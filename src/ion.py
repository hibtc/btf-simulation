


from pint import Quantity as Qty
c_light = Qty('c')



class Ion:
    """Class representing an ion

    

    Attributes:
        q: Charge
        m: Rest mass
        p: Momentum
        
        Brho: Magnetic rigidity
        
    """
    def __init__(self, name, q, m, Ekin_per_u):
        self.name = name
        self.q = Qty(q)
        self.m = Qty(m)
        self.p = ( (self.E0 + Qty(Ekin_per_u) * self.m)**2 - self.E0**2 )**0.5 / c_light
    
    @property
    def E0(self):
        """Rest energy"""
        return (self.m*c_light**2).to('MeV')
    @property
    def E(self):
        """Total energy"""
        return (( (self.p*c_light)**2 + self.E0**2 )**.5).to('MeV')
    @property
    def Ekin(self):
        """Kinetic energy"""
        return self.E - self.E0
    @property
    def Ekin_per_u(self):
        """Specific kinetic energy"""
        return (self.Ekin/self.m).to('MeV/u')
    @property
    def beta(self):
        """Relativistic beta"""
        return self.p*c_light/self.E
    @property
    def v(self):
        """Speed"""
        return self.beta * c_light
    @property
    def gamma(self):
        """Relativistic lorentz factor gamma"""
        return self.E/self.E0
    @property
    def Brho(self):
        """Magnetic rigidity"""
        return (self.p/self.q).to('T*m')
    @property
    def Erho(self):
        """Electric rigidity"""
        return (self.p*self.v/self.q).to('GV')
    def __repr__(self):
        return (f'Ion({self.name}, q={self.q:~Pg}, m={self.m:~Pg}, Ekin={self.Ekin_per_u:~Pg},\n'
                f'    beta={self.beta:~Pg}, gamma={self.gamma:~Pg}, Brho={self.Brho:~Pg}, Erho={self.Erho:~Pg})')
    
