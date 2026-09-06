"""Focused numerical contracts for functional-state dynamics."""
import io
import torch
from model import FitnessWorld


def main():
    torch.set_num_threads(2);torch.manual_seed(731);m=FitnessWorld().eval()
    c=torch.randn(4,128);g=torch.randn(4,930);q=torch.randn(4,930)
    s=m.encode(c,torch.tensor([0,1,0,1]));before=s.latent.clone();a=m.action(g);b=m.action(q)
    zero=m.intervene(s,a,torch.zeros(4,1,dtype=torch.bool))
    assert torch.equal(zero.latent,s.latent)
    empty=m.intervene(s,a[:,None,:][:,:0]);assert torch.equal(empty.latent,s.latent)
    after=m.intervene(s,a);assert torch.equal(s.latent,before)
    assert not torch.equal(after.latent,s.latent)
    p=m.observe(after,b);baseline=m.observe(s,b)
    assert torch.isfinite(p).all() and not torch.equal(p,baseline)
    p.square().mean().backward()
    assert m.transition[0].weight.grad.abs().sum()>0
    buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0)
    restored=FitnessWorld().eval();restored.load_state_dict(torch.load(buffer,weights_only=True))
    actual=restored.conditional(restored.encode(c,s.assay),restored.action(g),restored.action(q))
    assert torch.equal(p,actual)
    print('PASS: immutable state, masked/empty identity, conditional response, transition gradients, exact serialization')
    from capacity_model import FitnessWorld as CapacityWorld
    m=CapacityWorld().eval();s=m.encode(c,torch.tensor([0,1,0,1]));a=m.action(g);b=m.action(q)
    sa=m.intervene(s,a);sb=m.intervene(s,b);sab=m.intervene(sa,b);sba=m.intervene(sb,a)
    total=m.decode(sab,s);path=m.observe(s,a)+m.observe(sa,b)
    assert torch.allclose(total,path,atol=1e-6)
    assert torch.allclose(total,m.decode(sba,s),atol=1e-6)
    residual=total-m.observe(s,a)-m.observe(s,b)
    assert residual.abs().max()>1e-6
    residual.square().mean().backward();assert m.curvature.grad.abs().sum()>0
    assert torch.equal(m.decode(s,s),torch.zeros(len(c)))
    print('PASS: one viability landscape, exact path composition, symmetric double endpoint, nonlinear interaction, curvature gradients')


if __name__=='__main__':main()
