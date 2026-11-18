import numpy as np

# Her er regndataene for de fire boksene i teksten over

Ned_rad1 = np.array([40, 40, 85, 65])
vekt_orig_celler = np.array([0.925, 0.85, 0.85, 0.925])
vekt_overlapp    = np.array([0.075, 0.075, 0.075])

#a) Lag en np.array (av lengde 3) som inneholder regndataene for de tre overlappene
#   Legg svaret i variabelen svar_a

# Fyll inn din kode her

svar_a = (Ned_rad1[:-1] + Ned_rad1[1:])/2

# b) Lag Python kode som beregner gjennomsnittlig regnmengde. 
#    Legg svaret i variabelen svar_b

# Fyll inn din kode her
total_vekt = vekt_orig_celler.sum() + vekt_overlapp.sum()
svar_b = (Ned_rad1 @ vekt_orig_celler + svar_a @ vekt_overlapp)/total_vekt

print(svar_a, svar_b)