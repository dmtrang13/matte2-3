from scipy.integrate import tplquad

# integrand f(z, y, x) (merk rekkefølgen)
f = lambda z, y, x: z

# grenser: z: 0 -> x*y ; y: 0 -> x ; x: 0 -> 1
res, err = tplquad(
    func=f,
    a=0.0, b=1.0,                          # x
    gfun=lambda x: 0.0,
    hfun=lambda x: x,                      # y ∈ [0, x]
    qfun=lambda y, x: 0.0,
    rfun=lambda y, x: x*y                  # z ∈ [0, x*y]
)

print(res, err)