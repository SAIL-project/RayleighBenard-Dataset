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


# TODO MS: not sure if we always need this whole functionality, it really depends on the
# type of physical constraint that we assume.
class Tfunc:

    def __init__(
        self,
        segments,
        domain,
        action_limit,
        bcT_avg,
        x,
        fraction_length_smoothing=0.1,
    ):
        """
        segments: number of actuators/segments on the hot boundary layer
        domain: physical domain in both coordinates, horizontal direction last
        action_limit: this is the maximum fluctuation around the mean Tb that can be applied
        fraction_length_smoothing: the fraction of the cell that is used for smoothing
            (both ends have this fraction)
        """

        self.segments = segments
        self.domain = domain
        self.bcT_avg = bcT_avg
        self.x = x
        self.last_input = None
        self.last_action = None

        # Amplitude of variation of T
        self.ampl = action_limit

        self.xmax = float(
            self.domain[1][1]
        )

        # half-length of the interval on which we do the smoothing
        self.dx = 0.5 * fraction_length_smoothing * self.xmax / segments
        # self.dx = 0.03

    def apply_T(self, temperature_segments):
        """
        Centres the input around the bottom wall mean temperature Tb_avg preforms scaling to ensure the fluctuations don't exceed self.ampl
        apply_T: current implementation does the following:
        if fluctuations around mean are larger than 1:
            the max value of in the output with be self.ampl and the rest proportionally
        else:
            The temperature profile will just be scaled by self.ampl
        Cubic smoothing is applied between cells.
        """
        if np.array_equal(self.last_input, temperature_segments):
            return self.last_action

        x = self.x  # Sympy symbolic variable that will be the variable in the piecewise function which is returned
        Tb_avg = self.bcT_avg[0]
        values = self.ampl * temperature_segments
        Mean = values.mean()
        K2 = max(1, np.abs(values - np.array([Mean] * self.segments)).max() / self.ampl)
        xmax = self.xmax
        seq = []
        i = 0
        while i < self.segments - 1:  # Temperatures will vary between: Tb +- self.ampl

            x0 = i * xmax / self.segments  # physical value of the left bound of the segment
            x1 = (i + 1) * xmax / self.segments  # physical value of the right bound of the segment

            T1 = Tb_avg + (self.ampl * temperature_segments[i] - Mean) / K2
            T2 = Tb_avg + (self.ampl * temperature_segments[i + 1] - Mean) / K2
            # MS: periodic boundary conditions?
            if i == 0:
                T0 = Tb_avg + (self.ampl * temperature_segments[self.segments - 1] - Mean) / K2
            else:
                T0 = Tb_avg + (self.ampl * temperature_segments[i - 1] - Mean) / K2

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

            i += 1

            if i == self.segments - 1:
                x0 = i * xmax / self.segments
                x1 = (i + 1) * xmax / self.segments
                T0 = Tb_avg + (self.ampl * temperature_segments[i - 1] - Mean) / K2
                T1 = Tb_avg + (self.ampl * temperature_segments[i] - Mean) / K2
                T2 = Tb_avg + (self.ampl * temperature_segments[0] - Mean) / K2

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
        self.last_action = sympy.Piecewise(*seq)
        return self.last_action


    def get_last_action_numeric(self):
        """
        Returns the last action as a list of numeric values, for each heating segment one value.
        The heating segments are evaluated in the middle part of their domain. 
        """
        return [float(self.last_action.evalf(subs={self.x: self.xmax / self.segments * (i + 0.5)})) for i in range(self.segments)]