#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# Tfunc.py: defines temperature profiles on bottom wall that are fed to agent for actuation
#
# Colin Vignon
#
# FLOW, KTH Stockholm | 09/04/2023
import numpy as np
import sympy


class Tfunc:

    def __init__(self, nb_seg, domain, action_scaling):
        """N = number of actuators/segments on the hot boundary layer
        dicTemp = temperature variations of the segments: Ti' = Tnormal + Ti, Tnormal = 0.6 here"""

        self.nb_seg = nb_seg
        self.domain = domain

        # Amplitude of variation of T
        self.ampl = action_scaling

        # half-length of the interval on which we do the smoothing
        self.dx = 0.03

    def apply_T(self, dicTemp, x, bcT_avg):
        Tb_avg = bcT_avg[0]
        values = self.ampl * np.array(list(dicTemp.values()))
        Mean = values.mean()
        # TODO find out what K2 is?
        K2 = max(1, np.abs(values - np.array([Mean] * self.nb_seg)).max() / self.ampl)

        # Position:
        xmax = float(self.domain[1][1])    # TODO xmax was not a numerical value here, is that intended?
        # ind = sympy.floor(self.nb_seg * x // xmax)

        seq = []
        count = 0
        while count < self.nb_seg - 1:  # Temperatures will vary between: Tb +- 0.75

            x0 = count * xmax / self.nb_seg     # TODO "x0" and "x1" should not be symbolic...
            x1 = (count + 1) * xmax / self.nb_seg

            T1 = Tb_avg + (self.ampl * dicTemp.get("T" + str(count)) - Mean) / K2
            T2 = Tb_avg + (self.ampl * dicTemp.get("T" + str(count + 1)) - Mean) / K2
            if count == 0:
                T0 = Tb_avg + (self.ampl * dicTemp.get("T" + str(self.nb_seg - 1)) - Mean) / K2
            else:
                T0 = Tb_avg + (self.ampl * dicTemp.get("T" + str(count - 1)) - Mean) / K2

            seq.append(
                (
                    T0
                    + ((T0 - T1) / (4 * self.dx**3))
                    * (x - x0 - 2 * self.dx)
                    * (x - x0 + self.dx) ** 2,
                    x < x0 + self.dx,
                )
            )  # cubic smoothing
            seq.append((T1, x < x1 - self.dx))
            seq.append(
                (
                    T1
                    + ((T1 - T2) / (4 * self.dx**3))
                    * (x - x1 - 2 * self.dx)
                    * (x - x1 + self.dx) ** 2,
                    x < x1,
                )
            )  # cubic smoothing

            count += 1

            if count == self.nb_seg - 1:
                x0 = count * xmax / self.nb_seg
                x1 = (count + 1) * xmax / self.nb_seg
                T0 = Tb_avg + (self.ampl * dicTemp.get("T" + str(count - 1)) - Mean) / K2
                T1 = Tb_avg + (self.ampl * dicTemp.get("T" + str(count)) - Mean) / K2
                T2 = Tb_avg + (self.ampl * dicTemp.get("T0") - Mean) / K2

                seq.append(
                    (
                        T0
                        + ((T0 - T1) / (4 * self.dx**3))
                        * (x - x0 - 2 * self.dx)
                        * (x - x0 + self.dx) ** 2,
                        x < x0 + self.dx,
                    )
                )
                seq.append((T1, x < x1 - self.dx))
                seq.append(
                    (
                        T1
                        + ((T1 - T2) / (4 * self.dx**3))
                        * (x - x1 - 2 * self.dx)
                        * (x - x1 + self.dx) ** 2,
                        True,
                    )
                )

        return sympy.Piecewise(*seq)
