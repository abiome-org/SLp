# SLp-1.1 static entity universe v1

This self-contained OMF module turns two already-normalized, outcome-blind
proteome identity snapshots into one deterministic species-aware token
universe. It preserves SGD action identities, UniProtKB readout-query
identities, and every typed one-to-one or one-to-many relation without using a
held roster or any quantitative field. The relation-closed universe has 7,037
entities: 6,326 current model-eligible keys plus 711 SGD relation-support-only
genes that are not promoted to actions.

This v1 artifact is yeast-only. Its composite identity contract is the input
boundary for the planned corpus-v1.2 migration; the historical sparse-world
consumer still keys by bare entity ID and must not be used as evidence of
multi-species support.

The production input identities are compiled into the module. The workload
template names those immutable DatasetSnapshot revisions directly. Running it
would create a new identity artifact and therefore still requires the normal
clean-Git, rights, validation, and admission protocol; this repository change
does not run or admit that biological artifact.

See `CONTRACT.md` for the normative byte and trust boundary.
