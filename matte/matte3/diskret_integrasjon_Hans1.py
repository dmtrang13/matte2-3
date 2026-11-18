import numpy as np

matrise = np.array([[40, 40, 85, 65],[50, 75, 90, 20]]) #her er matrisen med verdier som skal brukes

#a) Beregn verdier til horisontale og vertikale overlappsceller
# Bruk variablene under (første som navn sier for horisontal overlapp, neste for vertikal)

nyceller_horisontal = (matrise[:, :-1] + matrise[:, 1:])/2
nyceller_vertikal = np.mean(matrise, axis=0, keepdims = True)

# b) Lag 3 matriser med vektene til cellene, en matrise for de originale cellene, en for vertikal overlapp og en for horisontal overlapp. 
# Regn også ut summen av vektene og legg svaret i variablen sum_vekt
v_overlap = 0.075
v_hjorne = 1-2*v_overlap  
v_tre_overlap = 1-3*v_overlap

vekt_orig_cell = np.array([[v_hjorne, v_tre_overlap, v_tre_overlap, v_hjorne],
                           [v_hjorne, v_tre_overlap, v_tre_overlap, v_hjorne]],
                          dtype=float)
vekt_hor_overlapp = np.full_like(nyceller_horisontal, v_overlap, dtype=float)
vekt_vert_overlapp = np.full_like(nyceller_vertikal, v_overlap, dtype=float)

sum_vekt = vekt_orig_cell.sum() + vekt_hor_overlapp.sum() + vekt_vert_overlapp.sum()

#c) Bruk svarene fra a) og b) til å beregne gjennomsnittlig verdi i dette eksemplet med to rader. Bruk formelen gitt i teksten.
#   Legg resultatet i variabelen gjennomsnitt_to_rad
teller = (matrise * vekt_orig_cell).sum() \
       + (nyceller_horisontal * vekt_hor_overlapp).sum() \
       + (nyceller_vertikal   * vekt_vert_overlapp).sum()
gjennomsnitt_to_rad = teller/sum_vekt