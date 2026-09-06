"""Focused scientific interface checks using generated, non-biological tensors."""
import unittest,json,tempfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from model import CellWorld,Config
from inference import WorldModel


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(731)
        self.model=CellWorld(Config(width=32,slots=4,heads=4,encoder_layers=1,dynamics_layers=1,flow_layers=1)).eval()
        self.values=torch.randn(3,11);self.basal=torch.randn(3,11);self.q=torch.randn(11,702);self.mod=torch.zeros(11,dtype=torch.long)
        self.mask=torch.ones(3,11,dtype=torch.bool);self.ctx=(torch.zeros(3,dtype=torch.long),)*3
        self.actions=torch.randn(3,2,642);self.am=torch.ones(3,2,dtype=torch.bool)
        self.state=self.model.encode(self.values,self.basal,self.q,self.mod,self.mask,*self.ctx)
        # Exercise a nontrivial transition, beyond its identity initialization.
        with torch.no_grad():self.model.delta.weight.normal_(0,.01);self.model.velocity.weight.normal_(0,.01)

    def test_exchangeable_query_observations(self):
        p=torch.randperm(11)
        perm=self.model.encode(self.values[:,p],self.basal[:,p],self.q[p],self.mod[p],self.mask[:,p],*self.ctx)
        torch.testing.assert_close(perm,self.state,atol=2e-6,rtol=2e-6)

    def test_exchangeable_actions_and_padding(self):
        reference=self.model.transition(self.state,self.actions,self.am,*self.ctx)
        reverse=self.model.transition(self.state,self.actions.flip(1),self.am,*self.ctx)
        torch.testing.assert_close(reference,reverse,atol=1e-6,rtol=1e-6)
        actions=torch.cat((self.actions,torch.full((3,3,642),float('nan'))),1)
        mask=torch.cat((self.am,torch.zeros(3,3,dtype=torch.bool)),1)
        torch.testing.assert_close(reference,self.model.transition(self.state,actions,mask,*self.ctx),atol=1e-6,rtol=1e-6)

    def test_empty_action_is_identity(self):
        mask=torch.zeros_like(self.am)
        self.assertTrue(torch.equal(self.state,self.model.transition(self.state,self.actions,mask,*self.ctx)))
        self.assertTrue(torch.equal(self.state,self.model.sample_state(self.state,self.actions,mask,*self.ctx,steps=2)))
        empty=self.model.transition(self.state,self.actions[:,:0],mask[:,:0],*self.ctx)
        self.assertTrue(torch.equal(empty,self.state))

    def test_masks_hide_missing_values(self):
        mask=self.mask.clone();mask[:,-2:]=False
        ref=self.model.encode(self.values,self.basal,self.q,self.mod,mask,*self.ctx)
        values=self.values.clone();basal=self.basal.clone();values[:,-2:]=float('nan');basal[:,-2:]=float('nan')
        torch.testing.assert_close(ref,self.model.encode(values,basal,self.q,self.mod,mask,*self.ctx))
        values[:,0]=float('nan')
        with self.assertRaises(ValueError):self.model.encode(values,basal,self.q,self.mod,mask,*self.ctx)

    def test_both_actions_and_state_receive_gradients(self):
        actions=self.actions.clone().requires_grad_();state=self.state.detach().requires_grad_()
        changed=self.model.transition(state,actions,self.am,*self.ctx)
        values,_=self.model.decode(changed,self.q,self.mod,*self.ctx);values.square().sum().backward()
        self.assertGreater(float(actions.grad[:,0].abs().sum()),0)
        self.assertGreater(float(actions.grad[:,1].abs().sum()),0)
        self.assertGreater(float(state.grad.abs().sum()),0)

    def test_generation_is_conditional_and_replayable(self):
        noise=torch.randn_like(self.state)*.3
        a=self.model.sample_state(self.state,self.actions,self.am,*self.ctx,noise=noise,steps=3)
        b=self.model.sample_state(self.state,self.actions,self.am,*self.ctx,noise=noise,steps=3)
        self.assertTrue(torch.equal(a,b))
        c=self.model.sample_state(self.state,self.actions,self.am,*self.ctx,noise=-noise,steps=3)
        self.assertGreater(float((a-c).abs().max()),.01)
        with self.assertRaises(ValueError):self.model.sample_state(self.state,self.actions,self.am,*self.ctx,steps=0)

    def test_decoder_query_order_and_unseen_panels(self):
        p=torch.randperm(11);a=self.model.decode(self.state,self.q,self.mod,*self.ctx)[0]
        b=self.model.decode(self.state,self.q[p],self.mod[p],*self.ctx)[0]
        torch.testing.assert_close(a[:,p],b)
        result=self.model.decode(self.state,torch.randn(7,702),torch.zeros(7,dtype=torch.long),*self.ctx)[0]
        self.assertEqual(result.shape,(3,7));self.assertTrue(torch.isfinite(result).all())

    def test_sparse_observation_distribution_and_standalone_replay(self):
        config=Config(width=32,slots=4,heads=4,encoder_layers=1,dynamics_layers=1,flow_layers=1,observation_likelihood=True)
        model=CellWorld(config).eval()
        with tempfile.TemporaryDirectory(prefix='slp-cell-world-') as temporary:
            path=Path(temporary).resolve()
            self.assertTrue(path.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            save_file(model.state_dict(),str(path/'model.safetensors'))
            (path/'config.json').write_text(json.dumps(asdict(config)))
            np.savez(path/'normalizer.npz',mean=np.zeros(642,np.float32),scale=np.ones(642,np.float32))
            world=WorldModel(path)
            modality=np.zeros(11,np.int64);modality[-2:]=1
            observed=self.values.abs().numpy();basal=np.ones_like(observed)*.2
            state=world.encode(observed,basal,self.q.numpy(),modality=modality,scale=1.,assay=5,taxon=9606,mechanism=0)
            untouched=world.decode(state)['values'];np.testing.assert_array_equal(untouched,observed)
            kwargs=dict(seed=731,steps=2)
            a=world.generate(state,self.actions.numpy(),self.am.numpy(),chunk_size=5,**kwargs)
            b=world.generate(state,self.actions.numpy(),self.am.numpy(),chunk_size=11,**kwargs)
            np.testing.assert_allclose(a['values'],b['values'],atol=1e-5,rtol=1e-5)
            self.assertTrue(np.isfinite(a['values']).all())
            self.assertTrue((a['values'][:,:9]>=0).all());self.assertTrue((a['values'][:,:9]<=np.log1p(10000)+1e-5).all())
            self.assertTrue((a['values'][:,:9]==0).any());self.assertTrue((a['values'][:,:9]>0).any())
            c=world.generate(state,self.actions.numpy(),self.am.numpy(),seed=732,steps=2)
            self.assertGreater(float(np.abs(a['values']-c['values']).max()),.01)
            with self.assertRaises(ValueError):world.encode(observed,basal,self.q.numpy(),modality=modality,scale=1.,assay=5,taxon=10090,mechanism=0)


if __name__=='__main__':unittest.main()
