# slant: SLant (Benstead-Hume et al. 2019, PLoS Comput Biol 15:e1006888)

battery: none; species: (human, yeasts, fly, worm in the paper); needs: unavailable code

- Method: random forest on PPI-network topology + GO features of gene pairs, trained on SL labels from five species (BioGRID/SynLethDB); predictions distributed through the Slorth database.
- Code: https://bitbucket.org/bioinformatics_lab_sussex/slant -> the Bitbucket workspace has been deactivated ("deactivated due to inactivity", API check 2026-09-24); no GitHub mirror found. Slorth web server (slorth.biochem.sussex.ac.uk) returns 403.
- Status: NOT acquired (code unavailable). Released Slorth predictions are handled by models-mechanistic (Slorth transfer), per board agreement.
- Leakage (if predictions were used): trained on SL labels across species => leaky.
