"""Simulate acceleration behavior of a stepper motor."""

import math

import numpy as np

from . import convert


class Accel:
    """Simulate acceleration behavior of a stepper motor."""

    def __init__(
        self,
        steps,
        steps_per_r,
        speed_max,
        accel_max,
        start_delay=0.0,
        stop_delay=0.0,
        si_units=False,
    ):
        """
        Calculate motor movement

        Values in rpm and rpm per second.

        Parameters
        ----------
        steps: int
            Number of full steps.
        steps_per_r: int
            Number of full steps per motor revolution.
        speed_max: float, optional
            Maximum motor speed in rpm.
            If no value is provided, it is searched in the config parameters.
        accel_max: float, optional
            Motor acceleration in rpm per second.
            If no value is provided, it is searched in the config parameters.
        start_delay, stop_delay: float, optional
            Delays in seconds before and after movement.
        si_units: bool, optional
            Indicate if speed_max and accel_max are in SI values rather than
            rpm and rpm per second.
        """
        # conversion to SI units
        if si_units:
            self.v_mot_max = speed_max
            self.a_mot_max = accel_max
        else:
            self.v_mot_max = convert.rpm_to_radpsec(speed_max)
            self.a_mot_max = convert.rpmps_to_radpsec2(accel_max)

        self.start_delay = start_delay
        self.stop_delay = stop_delay

        # no movement
        if steps == 0:
            self.t = [
                0,
                start_delay,
                start_delay + stop_delay,
            ]
            self.v = [
                0,
                0,
                0,
            ]

        # movement
        else:
            self.sign = steps / abs(steps)
            self.s_tot = abs(steps) / steps_per_r * 2 * math.pi

            # calculate max acceleration graph
            self.t_accel_max = self.v_mot_max / self.a_mot_max
            self.s_accel_max = self.a_mot_max * self.t_accel_max**2 / 2

            # case speed limit not reached
            # v ^
            #   |   /\
            #   |  /  \
            #   |_/____\__> t
            #
            if 2 * self.s_accel_max >= self.s_tot:
                self.s_accel = self.s_tot / 2
                self.t_accel = math.sqrt(2 * self.s_accel / self.a_mot_max)
                self.s_lin = 0
                self.t_lin = 0
                self.v_max = self.a_mot_max * self.t_accel

            # case speed limit is reached
            # v ^    ____
            #   |   /    \
            #   |  /      \
            #   |_/________\__> t
            #
            else:
                self.s_accel = self.s_accel_max
                self.t_accel = self.t_accel_max
                self.s_lin = self.s_tot - 2 * self.s_accel
                self.t_lin = self.s_lin / self.v_mot_max
                self.v_max = self.v_mot_max

            # compute graph characteristics
            self.t_tot = 2 * self.t_accel + self.t_lin
            t = self.start_delay
            self.t = [
                0,
                t,
                t + self.t_accel,
                t + self.t_accel + self.t_lin,
                t + 2 * self.t_accel + self.t_lin,
                t + 2 * self.t_accel + self.t_lin + self.stop_delay,
            ]
            self.v = [
                0,
                0,
                self.sign * self.v_max,
                self.sign * self.v_max,
                0,
                0,
            ]

        self.t = np.array(self.t)
        self.t_tot = self.t[-1]

        self.v_si = np.array(self.v)
        self.v_rpm = convert.radpsec_to_rpm(self.v_si)
