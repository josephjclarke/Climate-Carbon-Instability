#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import dataclasses
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"


@dataclasses.dataclass(kw_only=True)
class ClimateCarbonSystem:
    """
    Climate carbon system with a BEAM ocean.

    Most parameters can be edited via the constructor.
    """

    Ca0: float = 286.085 * 2.12
    CL0: float = 1630
    V1V2ratio: float = 1 / 50.0
    nu1: float = 0.2
    nu2: float = 1 / 500.0
    kappa: float = 0.57
    Q2x: float = 3.7
    Gamma1: float = 0.13
    Gamma2: float = 0.01
    npp0: float = 65.0
    Ca05: float = 344 * 2.12
    Q10: float = 2.0
    L: float = 1.0
    t_final: float = 1e6
    AM: float = 1.77e20
    OM: float = 7.8e22
    Alk: float = 5130
    k1: float = 1e-6
    k2: float = 7.53e-10
    kH0: float = 1.91e3 * 0.995268992324945
    linear_ocean: bool = False
    burnt_fossil_carbon: float = 0.0
    quadratic_feedback: float = 0.0
    cubic_feedback: float = 0.0
    land_global_warming_ratio: float = 1.3  # warming over land / warming globally
    sigma_Q: float = 0.5

    def __post_init__(self):
        """
        Calculate parameters from default values.

        Returns
        -------
        None.

        """
        A = self.calc_A()
        self.alpha = 0.1 * np.log(self.Q10) * self.land_global_warming_ratio
        self.dnppdCa0 = self.dnppdCa(self.Ca0)
        self.C10 = (
            4 * A * self.Alk * self.k2
            + np.sqrt(self.Ca0 * self.k1)
            * np.sqrt(self.Ca0 * self.k1 + 8 * A * self.Alk * self.k2)
            - self.Ca0 * (self.k1 - 8 * self.k2)
        ) / (8 * A * self.k2)
        self.C20 = self.C10 / self.V1V2ratio
        self.pi_equilibrium = np.array([0.0, 0.0, self.CL0, self.C10, self.C20])
        dCL = -100.0
        dC1 = -100.0
        dC2 = -100.0
        self.dy = np.array([0.2, 0.2, dCL, dC1, dC2])
        self.y0 = self.pi_equilibrium + self.dy
        self.pi_carbon = self.Ca0 + self.CL0 + self.C10 + self.C20

    def warming_at_CO2(self, Q=None):
        """
        Calculate warming at elevated CO2.

        Returns
        -------
        float
            ECS.

        """
        if Q is None:
            Q = self.Q2x

        def eqm(T):
            lin = self.L * T
            quad = self.quadratic_feedback * T**2
            cube = self.cubic_feedback * T**3
            return lin + quad + cube - Q

        def eqmp(T):
            lin = self.L
            quad = 2 * self.quadratic_feedback * T
            cube = 3 * self.cubic_feedback * T**2
            return lin + quad + cube

        def eqmpp(T):
            return 2 * self.quadratic_feedback + 6 * self.cubic_feedback * T

        p = np.polynomial.Polynomial(
            [-Q, self.L, self.quadratic_feedback, self.cubic_feedback]
        )
        roots = p.roots()
        rr = roots[np.isreal(roots)]
        if rr.size == 0:
            return np.nan
        else:
            return rr.min() if self.quadratic_feedback < 0 else rr.max()

    def npp(self, Ca):
        """
        NPP as a function of CO2.

        Parameters
        ----------
        Ca : array_like
            CO2 PgC.

        Returns
        -------
        array_like
            NPP PgC/yr.

        """
        return self.npp0 * (Ca / self.Ca0) * (self.Ca0 + self.Ca05) / (Ca + self.Ca05)

    def dnppdCa(self, Ca):
        """
        Devivative of NPP wrt CO2.

        Parameters
        ----------
        Ca : array_like
            CO2 PgC.

        Returns
        -------
        array_like
            dNPP/dCO2 /yr.

        """
        return (
            self.npp0
            * (self.Ca05 / self.Ca0)
            * (self.Ca0 + self.Ca05)
            / (Ca + self.Ca05) ** 2
        )

    def HenryCoeff(self, T=0):
        """
        Calculate ratio of molar concs of atm of ocean CO2.

        Can be temperature dependent in principal but here it is constant.

        Parameters
        ----------
        T : array_like, optional
            Ocean Temperature. The default is 0.

        Returns
        -------
        array_like
            array of kh.

        """
        return np.full_like(T, self.kH0, dtype=np.float64)

    def calc_A(self, T=0):
        """
        Evaluate the parameter A.

        A can depend on T but it does not here.

        Parameters
        ----------
        T : array_like, optional
            A is in principal temperature dependent. The default is 0.

        Returns
        -------
        array_like
            A.

        """
        return self.HenryCoeff(T) * self.AM / (self.OM / (1 + 1 / self.V1V2ratio))

    def calc_B(self, C1):
        """
        Calculate B.

        Parameters
        ----------
        C1 : array_like
            Upper ocean carbon PgC.

        Returns
        -------
        array_like
            B parameter.

        """
        c = np.full_like(C1, self.C10) if self.linear_ocean else C1

        B_b = (-self.Alk + c) * self.k1 + 4 * (self.Alk - 2 * c) * self.k2
        B_disc = (self.k1 * (self.Alk - c)) ** 2 - 4 * self.Alk * self.k1 * self.k2 * (
            self.Alk - 2 * c
        )
        return (B_b + np.sqrt(B_disc)) / (2 * c * (self.k1 - 4 * self.k2))

    def calc_dBdC1(self, C1):
        """
        Calculate derivative of B wrt C1.

        Parameters
        ----------
        C1 : array_like
            Upper ocean carbon PgC.

        Returns
        -------
        array_like
            derivative.

        """

        if self.linear_ocean:
            return np.zeros_like(C1)
        root = (self.k1 * (self.Alk - C1)) ** 2 - 4 * self.Alk * self.k1 * self.k2 * (
            self.Alk - 2 * C1
        )
        return (
            self.Alk * (1 + self.k1 * (C1 - self.Alk) / np.sqrt(root)) / (2 * C1**2)
        )

    def get_Ca(self, y):
        """
        Calculate atmospheric carbon.

        Parameters
        ----------
        y : array_like
            state vector.

        Returns
        -------
        array_like
            Atmospheric Carbon PgC.

        """
        T1, T2, CL, C1, C2 = y
        Ca = self.pi_carbon - CL - C1 - C2 + self.burnt_fossil_carbon
        return Ca

    def dFdt(self, t, y):
        """
        Equation of motion for climate carbon system.

        Parameters
        ----------
        t : float
            time.
        y : array_like
            State vector of length 5.

        Returns
        -------
        array_like
            State vector of length 5.

        """
        T1, T2, CL, C1, C2 = y
        Ca = self.get_Ca(y)

        A = self.calc_A()
        B = self.calc_B(C1)

        dOLR = self.L + self.quadratic_feedback * T1 + self.cubic_feedback * T1**2

        dT1 = self.Gamma1 * (
            -dOLR * T1
            + self.kappa * (T2 - T1)
            + (self.Q2x / np.log(2.0)) * np.log(Ca / self.Ca0)
        )
        dT2 = self.Gamma2 * (self.kappa * (T1 - T2))
        dCL = self.npp(Ca) - self.npp0 * (CL / self.CL0) * np.exp(self.alpha * T1)
        dC1 = self.nu1 * (Ca - A * B * C1) - self.nu2 * (C1 - self.V1V2ratio * C2)
        dC2 = self.nu2 * (C1 - self.V1V2ratio * C2)

        return np.asarray([dT1, dT2, dCL, dC1, dC2])

    def DdFdt(self, t, y):
        """
        Jacobian of dFdt.

        Parameters
        ----------
        t : float
            time.
        y : array_like
            State vector.

        Returns
        -------
        jac : array_like
            5x5 jacobian matrix.

        """
        T1, T2, CL, C1, C2 = y
        Ca = self.get_Ca(y)

        A = self.calc_A()
        B = self.calc_B(C1)
        dBdC1 = self.calc_dBdC1(C1)

        jac = np.zeros((y.size, y.size))

        ddOLR = 2 * self.quadratic_feedback * T1 + 3 * self.cubic_feedback * T1**2

        jac[0, 0] = -self.Gamma1 * (self.L + self.kappa + ddOLR)
        jac[0, 1] = self.Gamma1 * self.kappa
        jac[0, 2] = -self.Gamma1 * ((self.Q2x / np.log(2)) / Ca)
        jac[0, 3] = -self.Gamma1 * ((self.Q2x / np.log(2)) / Ca)
        jac[0, 4] = -self.Gamma1 * ((self.Q2x / np.log(2)) / Ca)

        jac[1, 0] = self.Gamma2 * self.kappa
        jac[1, 1] = -self.Gamma2 * self.kappa

        jac[2, 0] = -self.npp0 * (CL / self.CL0) * self.alpha * np.exp(self.alpha * T1)
        jac[2, 2] = -self.dnppdCa(Ca) - (self.npp0 / self.CL0) * np.exp(self.alpha * T1)
        jac[2, 3] = -self.dnppdCa(Ca)
        jac[2, 4] = -self.dnppdCa(Ca)

        jac[3, 2] = -self.nu1
        jac[3, 3] = self.nu1 * (-1 - A * B - A * dBdC1 * C1) - self.nu2
        jac[3, 4] = -self.nu1 + self.nu2 * self.V1V2ratio

        jac[4, 3] = self.nu2
        jac[4, 4] = -self.nu2 * self.V1V2ratio

        return jac

    def test_equilibrium(self, guess=None, verbose=True):
        """
        Checks that the equilibrium is an equilibrium.

        Parameters
        ----------
        guess   : array_like, optional
            Possible Equilibrium. Defaults to pi equilibrium.
        verbose : Bool, optional
            Print out result. Defaults to True

        Returns
        -------
        Bool.
            Equilibrium == Equilibrium.

        """
        eq = self.pi_equilibrium if guess is None else guess
        equilibrium_rate = self.dFdt(0.0, eq)
        if verbose:
            print(f"dFdt at equilibrium: {equilibrium_rate}")
        if np.allclose(equilibrium_rate, 0.0):
            if verbose:
                print("Analytic PI Equilibrium OK")
            return True
        else:
            if verbose:
                print("Analytic PI Equilibrium not OK")
            return False

    def integrate(self, initial_condition=None):
        """
        Solves the equation of motion.

        Returns
        -------
        None.

        """
        y0 = self.y0 if initial_condition is None else initial_condition
        sol = scipy.integrate.solve_ivp(
            self.dFdt, (0.0, self.t_final), y0, jac=self.DdFdt, method="LSODA"
        )
        self.t = sol.t
        self.T1, self.T2, self.CL, self.C1, self.C2 = sol.y
        self.Ca = self.get_Ca(sol.y)
        return sol

    def find_equilibrium(self, guess=None):
        """
        Compute the equilibrium by integration.

        There is no guarantee of convergence. Furthmore, the 'equilibrium' may be
        a point on a limit cycle.

        Parameters
        ----------
        guess : array_like, optional
            Initial guess of equilibrium. The default is None.

        Returns
        -------
        None.

        """
        y0 = self.pi_equilibrium if guess is None else guess
        self.equilibrium = scipy.optimize.fsolve(
            lambda y: self.dFdt(0.0, y), y0, fprime=lambda y: self.DdFdt(0.0, y)
        )

    def jac_eig_at(self, y=None):
        """
        Calculate the Jacobian's eigenvalue at equilibrium y.

        Parameters
        ----------
        y : array_like
            State Vector. Defaults to pi equilibrium.

        Returns
        -------
        array_like
            Eigenvalues.

        """
        y = y if y is not None else self.pi_equilibrium
        return np.linalg.eigvals(self.DdFdt(0, y))

    def calc_f(self) -> float:
        """
        Calculate the f value.

        Returns
        -------
        float
            d(AB C_1)/dC1.

        """
        A = self.calc_A()
        B = self.calc_B(self.C10)
        dB = self.calc_dBdC1(self.C10)

        return A * B + A * dB * self.C10

    def analytic_bf(self, use_simple_approx=False) -> (float, float):
        """
        Estimate lambda of bifurcation.

        Parameters
        ----------
        use_simple_approx : bool
            Use the approximation accurate to ~10%.
            Default value: False

        Returns
        -------
        float
            The critical lambda and critical ECS.

        """
        A = self.calc_A()
        B = self.calc_B(self.C10)
        dB = self.calc_dBdC1(self.C10)

        f = A * B + A * dB * self.C10
        mu = (
            self.Ca0 * self.dnppdCa0 / self.npp0
            + self.Ca0
            * (
                1
                + 1 / f
                + self.nu2 / (self.Gamma2 * self.kappa * f)
                + self.CL0 * self.nu2 / self.npp0 / f
            )
            / self.CL0
        ) * (
            1
            + (self.Ca0 * self.nu2 / self.Gamma2)
            / (self.CL0 * f * self.alpha * self.Q2x)
        )
        crit_ECS = mu * np.log(2.0) / self.alpha
        crit_L = self.Q2x / crit_ECS
        if use_simple_approx:
            return crit_L, crit_ECS

        num = self.Q2x * self.alpha * self.CL0 * self.Gamma1 * (
            f * self.V1V2ratio * self.nu1 * self.nu2
            + self.kappa
            * self.Gamma2
            * (f * self.nu1 + (1 + self.V1V2ratio) * self.nu2)
        ) * self.npp0 - self.kappa * np.log(2) * self.Ca0 * (
            self.Gamma1 + self.Gamma2
        ) * self.nu1 * self.nu2 * (
            f * self.dnppdCa0 * self.V1V2ratio * self.CL0
            + (1 + self.V1V2ratio + f * self.V1V2ratio) * self.npp0
        )
        den = (
            np.log(2)
            * self.Ca0
            * self.Gamma1
            * (
                self.CL0
                * (
                    f * self.dnppdCa0 * self.V1V2ratio * self.nu1 * self.nu2
                    + self.kappa
                    * self.Gamma2
                    * (
                        f * self.dnppdCa0 * self.nu1
                        + (
                            self.dnppdCa0 * (1 + self.V1V2ratio)
                            + (1 + self.V1V2ratio + f * self.V1V2ratio) * self.nu1
                        )
                        * self.nu2
                    )
                )
                + (
                    (1 + self.V1V2ratio + f * self.V1V2ratio) * self.nu1 * self.nu2
                    + self.kappa
                    * self.Gamma2
                    * ((1 + f) * self.nu1 + (1 + self.V1V2ratio) * self.nu2)
                )
                * self.npp0
            )
        )
        crit_L = num / den
        crit_ECS = self.Q2x / crit_L
        return crit_L, crit_ECS

    def analytic_growth_rate(self, L):
        """
        Calculate the growth rate near the bifurcation point.

        The growth rate is given by (lambda + g_1) / (g_2 lambda + g_3).

        Parameters
        ----------
        L : float
            Climate feedback parameter.

        Returns
        -------
        Growth rate : float.

        """
        A = self.calc_A()
        B = self.calc_B(self.C10)
        dB = self.calc_dBdC1(self.C10)
        f = A * B + A * dB * self.C10

        r_1 = (
            -np.log(2)
            * self.Ca0
            * (
                self.CL0
                * (
                    self.dnppdCa0 * (1 + self.V1V2ratio) * self.kappa * self.nu2
                    + self.nu1
                    * (
                        self.dnppdCa0 * f * self.kappa
                        + (
                            self.kappa
                            + self.V1V2ratio * self.kappa
                            + f * self.V1V2ratio * self.kappa
                            + self.dnppdCa0 * f * self.V1V2ratio * 1 / self.Gamma2
                        )
                        * self.nu2
                    )
                )
                + (
                    (1 + self.V1V2ratio) * self.kappa * self.nu2
                    + self.nu1
                    * (
                        (1 + f) * self.kappa
                        + (1 + self.V1V2ratio + f * self.V1V2ratio)
                        * 1
                        / self.Gamma2
                        * self.nu2
                    )
                )
                * self.npp0
            )
        )
        r_2 = (
            self.alpha
            * self.kappa
            * self.CL0
            * self.Q2x
            * (f * self.nu1 + (1 + self.V1V2ratio) * self.nu2)
            * self.npp0
            - self.kappa
            * np.log(2)
            * 1
            / self.Gamma1
            * self.Ca0
            * self.nu1
            * self.nu2
            * (
                self.dnppdCa0 * f * self.V1V2ratio * self.CL0
                + (1 + self.V1V2ratio + f * self.V1V2ratio) * self.npp0
            )
            - 1
            / self.Gamma2
            * self.nu1
            * self.nu2
            * (
                -f * self.V1V2ratio * self.alpha * self.CL0 * self.Q2x * self.npp0
                + self.kappa
                * np.log(2)
                * self.Ca0
                * (
                    self.dnppdCa0 * f * self.V1V2ratio * self.CL0
                    + (1 + self.V1V2ratio + f * self.V1V2ratio) * self.npp0
                )
            )
        )
        r_3 = (
            np.log(4)
            * self.Ca0
            * (
                self.CL0
                * (
                    self.dnppdCa0 * self.kappa
                    + (1 + self.V1V2ratio)
                    * (self.kappa + self.dnppdCa0 * 1 / self.Gamma2)
                    * self.nu2
                    + self.nu1
                    * (
                        self.kappa
                        + f * self.kappa
                        + self.dnppdCa0 * f * 1 / self.Gamma2
                        + (1 + self.V1V2ratio + f * self.V1V2ratio)
                        * 1
                        / self.Gamma2
                        * self.nu2
                    )
                )
                + (
                    self.kappa
                    + 1
                    / self.Gamma2
                    * ((1 + f) * self.nu1 + (1 + self.V1V2ratio) * self.nu2)
                )
                * self.npp0
            )
        )
        r_4 = 2 * (
            self.kappa
            * np.log(2)
            * 1
            / self.Gamma2
            * self.Ca0
            * self.CL0
            * (
                self.dnppdCa0 * (1 + self.V1V2ratio) * self.nu2
                + self.nu1
                * (
                    self.dnppdCa0 * f
                    + (1 + self.V1V2ratio + f * self.V1V2ratio) * self.nu2
                )
            )
            + (
                self.kappa
                * np.log(2)
                * 1
                / self.Gamma2
                * self.Ca0
                * ((1 + f) * self.nu1 + (1 + self.V1V2ratio) * self.nu2)
                - self.alpha
                * self.CL0
                * self.Q2x
                * (
                    self.kappa
                    + 1 / self.Gamma2 * (f * self.nu1 + (1 + self.V1V2ratio) * self.nu2)
                )
            )
            * self.npp0
            + np.log(2)
            * 1
            / self.Gamma1
            * self.Ca0
            * (
                self.CL0
                * (
                    self.dnppdCa0 * f * self.kappa * self.nu1
                    + (
                        self.dnppdCa0 * (1 + self.V1V2ratio) * self.kappa
                        + (
                            self.kappa
                            + self.V1V2ratio * self.kappa
                            + f * self.V1V2ratio * self.kappa
                            + self.dnppdCa0 * f * self.V1V2ratio * 1 / self.Gamma2
                        )
                        * self.nu1
                    )
                    * self.nu2
                )
                + (
                    (1 + f) * self.kappa * self.nu1
                    + (
                        (1 + self.V1V2ratio) * self.kappa
                        + (1 + self.V1V2ratio + f * self.V1V2ratio)
                        * 1
                        / self.Gamma2
                        * self.nu1
                    )
                    * self.nu2
                )
                * self.npp0
            )
        )

        g_1 = r_2 / r_1
        g_2 = r_3 / r_1
        g_3 = r_4 / r_1
        return (L + g_1) / (g_2 * L + g_3)
