class Convert:

    def __init__(self):
         """Convert A to B."""
        pass

    @staticmethod
    def rpm_to_radpsec(rpm):
        """Convert rpm to radiant per second"""
        return rpm / 60 * 2 * math.pi

    @staticmethod
    def rpmps_to_radpsec2(rpmps):
        """Convert rpm per second to radiant per second**2"""
        return rpmps / 60 * 2 * math.pi

    @staticmethod
    def radpsec_to_rpm(radpsec):
        """Convert radiant per second to rpm"""
        return radpsec * 60 / 2 / math.pi

    @staticmethod
    def radpsec2_to_rpmps(radpsec2):
        """Convert radiant per second**2 to rpm per second"""
        return radpsec2 * 60 / 2 / math.pi
