import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import integrate

class PcmProperties():
    """
    This class allows us to assign the properties of eutectic organic phase
    change materials.
    Ex: component_A = PcmProperties(pcm_data_A)
        component_B = PcmProperties(pcm_data_B)
        pcm_data_A should be properites of A ["Name",'molar_mass', 'T_fus',
        'Del_H', 'cp_solid_a', 'cp_solid_b', 'cp_solid_c', 'cp_liquid_a',
        'cp_liquid_b', 'cp_liquid_c', 'cp_liquid_d]

        Where cp_solid_a represents heat cpacity cp = a + bT+cT**2
    """
    def __init__(self, data):
        self.name = data[0]
        self.molar_mass = data[1]
        self.fusion_T = data[2]+273.15
        self.heat_of_fusion = data[3]*data[1]
        self.cp_solid_a = data[4]
        self.cp_solid_b = data[5]
        self.cp_solid_c = data[6]
        self.cp_liquid_a = data[7]
        self.cp_liquid_b = data[8]
        self.cp_liquid_c = data[9]
        self.cp_liquid_d = data[10]


class EutecticMixture():
    """
    This class calculates the binary mixture properties of the  organic phase
    materials

    """

    def __init__(self,A,B):
        """
        parameter A: It is an instance of the class PcmProperties for component_A
                  B: It is an instance of the class PcmProperties for component_B
        """
        self.A = A
        self.B = B
        self.R = 8.314
        self.font1 = {'family':'serif','color':'blue','size':18}
        self.font2 = {'family':'serif','color':'darkred','size':12}

    def eutectic_properties(self):
        """
        This method calculate the eutect mole fraction of A and eutect temperature
        """
        self.mole_fraction_A = [i for i in np.arange(0.0005,0.9995,0.0001)]
        self.temperature_BA = list(map(lambda xA: (self.A.fusion_T*self.A.heat_of_fusion)/(self.A.heat_of_fusion-self.R*self.A.fusion_T*math.log(xA)),self.mole_fraction_A))
        self.temperature_AB = list(map(lambda xA: (self.B.fusion_T*self.B.heat_of_fusion)/(self.B.heat_of_fusion-self.R*self.B.fusion_T*math.log(1-xA)),self.mole_fraction_A))
        for j in range(len(self.temperature_BA)-1):
            err0 = self.temperature_AB[j]-self.temperature_BA[j]
            err1 = self.temperature_AB[j+1]-self.temperature_BA[j+1]
            if err0*err1<0:
                break
        self.TE = (self.temperature_AB[j]+self.temperature_BA[j])/2
        self.xE = j*0.0001
        return self.TE, self.xE


    def plot_temp_AB(self):
        """
        This function plot the liquidus lines
        """
        fig, ax = plt.subplots()

        ax.scatter(self.mole_fraction_A,self.temperature_AB,marker =".")
        ax.scatter(self.mole_fraction_A,self.temperature_BA,marker =".")
        # To plot lines

        min_value = min(self.temperature_AB+self.temperature_BA)
        ax.plot([0,self.xE,self.xE],[self.TE,self.TE,min_value])
        ax.set_title("Plots of liquidus lines",fontdict = self.font1)
        ax.annotate(f'(xE={self.xE}, TE={round(self.TE,2)})', xy=(self.xE, self.TE),xytext=(self.xE+0.2,self.TE-20),arrowprops=dict(facecolor='red', shrink=0.05))
        ax.set_xlabel("Mole fraction of A",fontdict = self.font2)
        ax.set_ylabel("Temperature ($T^oC$)",fontdict = self.font2)
        return fig


    def entropy(self):
        """This method calculate the total entropy"""
        ds1_integrand = lambda T: (self.B.cp_solid_a + self.B.cp_solid_b*T +self.B.cp_solid_c*T**2)/T
        self.ds1 = (1-self.xE)*integrate.quad(ds1_integrand,self.TE,self.B.fusion_T)[0]
        self.ds2 = (1-self.xE)*self.B.heat_of_fusion/self.B.fusion_T

        ds3_integrand = lambda T: (self.A.cp_solid_a + self.A.cp_solid_b*T + self.A.cp_solid_c*T**2)/T
        self.ds3 = self.xE*integrate.quad(ds3_integrand,self.TE,self.A.fusion_T)[0]
        self.ds4 = self.xE*self.A.heat_of_fusion/self.A.fusion_T
        ds5_integrand = lambda T: (self.A.cp_liquid_a+self.A.cp_liquid_b*T+self.A.cp_liquid_c*T**2+self.A.cp_liquid_d*T**3)/T
        self.ds5 = self.xE*integrate.quad(ds5_integrand,self.A.fusion_T,self.B.fusion_T)[0]
        self.ds6 = -self.R*(self.xE*math.log(self.xE) + (1-self.xE)*math.log(1-self.xE))
        ds7_integrand = lambda T: (self.xE*(self.A.cp_liquid_a+self.A.cp_liquid_b*T +self.A.cp_liquid_c*T**2+ self.A.cp_liquid_d*T**3)+ (1-self.xE)*(self.B.cp_liquid_a+self.B.cp_liquid_b*T +self.B.cp_liquid_c*T**2+ self.B.cp_liquid_d*T**3))/T
        self.ds7 = integrate.quad(ds7_integrand,self.B.fusion_T,self.TE)[0]
        self.ds_total = self.ds1 + self.ds2 + self.ds3 + self.ds4 + self.ds5 + self.ds6 + self.ds7
        return self.ds_total

    def enthalpy(self):
        """This method calculate the total entropy"""

        dh1_integrand = lambda T: (self.B.cp_solid_a + self.B.cp_solid_b*T +self.B.cp_solid_c*T**2)
        self.dh1 = (1-self.xE)*integrate.quad(dh1_integrand,self.TE,self.B.fusion_T)[0]
        self.dh2 = (1-self.xE)*self.B.heat_of_fusion
        dh3_integrand = lambda T: (self.A.cp_solid_a + self.A.cp_solid_b*T + self.A.cp_solid_c*T**2)
        self.dh3 = self.xE*integrate.quad(dh3_integrand,self.TE,self.A.fusion_T)[0]
        self.dh4 = self.xE*self.A.heat_of_fusion
        dh5_integrand = lambda T: (self.A.cp_liquid_a+self.A.cp_liquid_b*T+self.A.cp_liquid_c*T**2+self.A.cp_liquid_d*T**3)
        self.dh5 = self.xE*integrate.quad(dh5_integrand,self.A.fusion_T,self.B.fusion_T)[0]
        self.dh6 = 0
        dh7_integrand = lambda T: (self.xE*(self.A.cp_liquid_a+self.A.cp_liquid_b*T +self.A.cp_liquid_c*T**2+ self.A.cp_liquid_d*T**3)+ (1-self.xE)*(self.B.cp_liquid_a+self.B.cp_liquid_b*T +self.B.cp_liquid_c*T**2+ self.B.cp_liquid_d*T**3))
        self.dh7 = integrate.quad(dh7_integrand,self.B.fusion_T,self.TE)[0]
        self.dh_total = self.dh1 + self.dh2 + self.dh3 + self.dh4 + self.dh5 + self.dh6 + self.dh7
        return self.dh_total


    def plot_entropy(self):
        self.entropy()
        fig, ax = plt.subplots()
        x_entropy = ['$\Delta S1$','$\Delta S2$','$\Delta S3$','$\Delta S4$','$\Delta S5$','$\Delta S6$','$\Delta S7$']
        self.ds = [self.ds1, self.ds2,self.ds3,self.ds4,self.ds5,self.ds6,self.ds7]
        ax.bar(x_entropy,self.ds)
        ax.set_xlabel("Entropy change",fontdict = self.font2)
        ax.set_ylabel("J/mol.K",fontdict = self.font2)
        return fig


    def plot_enthalpy(self):
        self.enthalpy()
        fig, ax = plt.subplots()
        x_enthalpy = ['$\Delta H1$','$\Delta H2$','$\Delta H3$','$\Delta H4$','$\Delta H5$','$\Delta H6$','$\Delta H7$']
        self.dh = [self.dh1, self.dh2,self.dh3,self.dh4,self.dh5,self.dh6,self.dh7]
        ax.bar(x_enthalpy,self.dh)
        ax.set_xlabel("Enthalpy change",fontdict = self.font2)
        ax.set_ylabel("J/mol",fontdict = self.font2)
        return fig
