import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pymc3 as pm
import pymc3.stats as pms
import pystan as st

observed_n_a = np.array((24, 25, 31, 31, 22, 21, 26, 20, 16, 22))
observed_deaths = np.array((734, 516, 754, 877, 814, 362, 764, 809, 223, 1066))
observed_d_rate = np.array(
    (0.19, 0.12, 0.15, 0.16, 0.14, 0.06, 0.13, 0.13, 0.03, 0.15)
)

miles_e8_estimate = observed_deaths / observed_d_rate


def version_pymc3():

    print("(a) -- independent θ, not modelling exposure")
    with pm.Model():
        θ = pm.Gamma("θ", alpha=1, beta=1)
        pm.Poisson("n_a", mu=θ, observed=observed_n_a)
        trace = pm.sample(5000, tune=1000, cores=4)

    print("(a) 95% credible interval for θ:")
    print(pms.hpd(trace["θ"]))

    print("(b) -- independent θ, modelling exposure")
    with pm.Model():
        θ = pm.Gamma("θ", alpha=1, beta=1)
        pm.Poisson("n_a", mu=miles_e8_estimate * θ, observed=observed_n_a)
        trace = pm.sample(5000, tune=1000, cores=4)

    print("(b) 95% credible interval for θ:")
    print(pms.hpd(trace["θ"]))
    pm.traceplot(trace)
    plt.show()


def version_pystan():

    code = """
    data {
        int n_obs;
        int n_accidents[n_obs];
    }
    parameters {
        real<lower=0> rate;
    }
    model {
        real alpha = 1.0;
        real beta = 1.0;
        rate ~ gamma(alpha, beta);
        n_accidents ~ poisson(rate);
    }
    generated quantities {
        real pred;
        pred <- poisson_rng(rate);
    }
    """

    sm = st.StanModel(model_code=code)
    fit = sm.sampling(
        data={"n_obs": len(observed_n_a), "n_accidents": observed_n_a},
        iter=5000,
        chains=4,
    )
    print(fit)


if __name__ == "__main__":
    # version_pymc3()
    version_pystan()
