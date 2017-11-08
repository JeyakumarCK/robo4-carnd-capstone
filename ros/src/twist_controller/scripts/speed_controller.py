from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class SpeedController(object):
    def __init__(self):
        kp_pos = 0.1
        kp_neg = 0.5
        ki = 0
        kd = 0
        # is_initialized = False # TODO Implement intialization (handle dt)
        self.pid = PID(kp_pos, kp_neg, ki, kd, min=-1, max=1)
        self.v_error_braking_threshold = 2 * 1.6 / 3.6
        self.v_min_braking_threshold = 5 * 1.6 / 3.6 # Speed below which braking threshold is disregarded

    def control(self, demand, x, time_step):
        error = demand - x
        val = self.pid.step(error, time_step)
        
        throttle = 0
        brake = 0

        if val > 0: 
            throttle = val
        else:
            # Only brake is the velocity error is greater than threshold
            if abs(error) > self.v_error_braking_threshold or demand < self.v_min_braking_threshold:
                brake = -val
        # When target velocity and actual velocity are close to zero,
        # set brake to constant value.
        if ((demand < abs(0.2)) and (x < abs(0.2))):
            # in dbw_node.py multiplying brake by 200, so brake constant value set to 3
            brake = max(3, brake)

        return throttle, brake

    def reset(self):
        self.pid.reset()
