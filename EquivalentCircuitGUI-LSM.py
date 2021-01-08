import cmath
import math
import tkinter
from tkinter import filedialog
from PyEIS import *
import lmfit

import gamry_parser as parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from galvani import BioLogic
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.optimize import curve_fit

##
#Created by Padraig Stack for use at The University of Akron
#Last Edited 3 January 2021 by Padraig Stack
#Contact: pis5@zips.uakron.edu
##

font = ("Helvetica", 14)
font2 = ("Helvetica", 12)
def Parallel_eq(element1, element2, element3=np.inf, element4 = np.inf, element5 = np.inf): #Parallel equivalent impedence
    inv_res = (1/element1) + (1/element2) + (1/element3) + (1/element4) + (1/element5)
    res = (1/inv_res)
    return res

def file_select(): #Allows the user to select EIS data to load in the calculator. Exports real and imaginary impedence
    filename = filedialog.askopenfilename(filetypes=[("BioLogic or Gamry", ".mpr .DTA")])
    if ".mpr" in filename:
        mpr_file = BioLogic.MPRfile(filename)
        df = pd.DataFrame(mpr_file.data)
        freq = df.iloc[:,0]
        real_imp = df.iloc[:,1]
        img_imp = df.iloc[:,2]
        Z = []
        for i in range(len(real_imp)):
            Z.append(complex(float(real_imp[i]), -1*float(img_imp[i])))
        Z = np.array(Z)
        freq = np.array(freq)        

    if ".DTA" in filename:
        freq, Z = preprocessing.readGamry(filename)
    return Z, freq

def plot_ri(data, *args, **kwargs):
        plt.plot(data.real, -data.imag, *args, **kwargs)

def simple_randles_eq(x, R0, CPE0_0, CPE0_1, R1):
        cpe_imp = 1/(float(CPE0_0)*(complex(0,1)*x)**float(CPE0_1))
        total_imp = (float(R0) + Parallel_eq(float(R1), cpe_imp))
        return total_imp

class SimpleRandlesModel(lmfit.model.Model):
        __doc__ = "Simple Randles Model" + lmfit.models.COMMON_DOC
        def __inint__(self, *args, **kwargs):
            super().__init__(simple_randles_eq, *args, **kwargs)
            self.set_param_hint('CPE0_1', min=0, max=1)
        
        def guess(self, data, x=None, **kwargs):
            verbose = kwargs.pop('verbose', None)
            if x is None:
                return
            argmin_x = np.abs(data).argmin()
            xmin = x.min()
            xmax = x.max()
            R0, CPE0_0, CPE0_1, R1 = [20, 5.37e-05, 8.68e-01, 4.20e+03]
            if verbose:
                print()
            params = self.make_params(x = x, R0 = R0, CPE0_0 = CPE0_0, CPE0_1 = CPE0_1, R1 = R1)
            return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

def cylindrical_pore(freq, Rs, Rm, L, Cw, Rw, Cu, Ru):
    CapW =  1/(freq*Cw*(complex(0,1)))
    Zw = Parallel_eq(CapW, Rw)
    Capu =  1/(freq*Cu*(complex(0,1)))
    Zu = Parallel_eq(Capu, Ru)
    #Zw = 1/(freq*Cw*(complex(0,1)))
    #Zu = 1/(freq*Cu*(complex(0,1)))
    #numerator = (2*Rm*Rs * np.sqrt((Rm+Rs)/Zw)) + (np.sqrt((Rm+Rs)/Zw) * ((Rm**2) + (Rs**2)) * np.cosh(L*np.sqrt((Rm+Rs)/Zw))) + ((Rs**2) * ((Rm+Rs)/Zu)*np.sinh(L*np.sqrt((Rm+Rs)/Zw)))
    #denominator = np.sqrt( (((Rm+Rs)/Zw)*(Rm+Rs)*( (np.sqrt((Rm+Rs)/Zw)*np.sinh(L*np.sqrt((Rm+Rs)/Zw))) + (((Rm+Rs)/Zu)*np.cosh(L*np.sqrt((Rm+Rs)/Zw))) ) ) )
    base_term_w = ((Rm+Rs)/Zw)
    base_term_u = ((Rm+Rs)/Zu)
    square_term_w = np.sqrt(base_term_w)
    numerator = (2*Rm*Rs*square_term_w) + (square_term_w*((Rm**2)+(Rs**2)) * np.cosh(L*square_term_w)) + ((Rs**2)*base_term_u)*np.sinh(L*square_term_w)
    denominator = np.sqrt( (base_term_w)*(Rm+Rs) * ( (square_term_w*np.sinh(L*square_term_w)) + (base_term_u) * np.cosh(L*square_term_w) ) )
    
    Zp = ((Rm*Rs*L)/(Rm+Rs)) + (numerator/denominator)
    return Zp

def cylindrical_pore2(freq, Rs, Rm, L, Cw, Rw, Cu, Ru):
    numerator = (2*np.sqrt(gamma)*Rm*Rs) + (np.sqrt(gamma)*(Rm**2+R))

    Zp = (Rm*Rs*L)/(Rm+Rs) + numerator/denominator
    return Zp
class CylindricalModel(lmfit.model.Model):
        __doc__ = "Park and Macdonald Cylindrical" + lmfit.models.COMMON_DOC
        def __inint__(self, *args, **kwargs):
            super().__init__(simple_randles_eq, *args, **kwargs)
            self.set_param_hint('CPE0_1', min=0, max=1)
        
        def guess(self, data, freq=None, **kwargs):
            verbose = kwargs.pop('verbose', None)
            Rs, Rm, L, Cw, Rw, Cu, Ru = [100e04, 13e4, 0.015, 800e-5, 40e2, 0.5e-6, 180e3]
            if verbose:
                print()
            params = self.make_params(freq = freq, Rs = Rs, Rm = Rm, L = L, Cw = Cw, Rw = Rw, Cu = Cu, Ru = Ru)
            return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

def transmission_line(freq, N):
    alpha = 0
    beta = 1
    xi = 1
    eta = 1
    Zu = 1
    
    numerator = xi - (Zu * alpha)
    denominator = eta - (Zu* beta)
    Zp = numerator/denominator
    return Zp

class TransmissionModel(lmfit.model.Model):
        __doc__ = "Eloot, et al Transmission" + lmfit.models.COMMON_DOC
        def __inint__(self, *args, **kwargs):
            super().__init__(simple_randles_eq, *args, **kwargs)
            self.set_param_hint('CPE0_1', min=0, max=1)
        
        def guess(self, data, freq=None, **kwargs):
            verbose = kwargs.pop('verbose', None)
            Rs, Rm, L, Cw, Rw, Cu, Ru = [100e04, 13e4, 0.015, 800e-5, 40e2, 0.5e-6, 180e3]
            if verbose:
                print()
            params = self.make_params(freq = freq, Rs = Rs, Rm = Rm, L = L, Cw = Cw, Rw = Rw, Cu = Cu, Ru = Ru)
            return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

class MainWindow:
    def __init__(self, master): #InitialMenu
        self.master = master
        master.title("Select Equivalent Circuit Method")
        master.configure(bg="gray69")
        tkinter.Label(master, text="Please select the following Equivalent Circuit Method you would like to use", font=(font)).grid(row=0, columnspan = 3)
        self.s_rand_pic = tkinter.PhotoImage(file="SimpRandles.png") #Images drawn on https://www.circuit-diagram.org/editor/
        self.warburg_pic = tkinter.PhotoImage(file="Warburg.png")
        self.psuedo_pic = tkinter.PhotoImage(file="Psuedocapacitance.png")
        self.five_transmission_pic = tkinter.PhotoImage(file="5TermTransmission.png")

        srand_btn = tkinter.Button(master, image = self.s_rand_pic, command = self.srand_calc_impedance)
        srand_btn.image = self.s_rand_pic
        srand_btn.grid(row=1,column=0)
        tkinter.Label(master, text="Simple Randles", font=font2).grid(row=2,column=0)

        warburg_btn = tkinter.Button(master, image = self.warburg_pic, command = self.warburg_calc_impedance)
        warburg_btn.image = self.warburg_pic
        warburg_btn.grid(row=1,column=1)
        tkinter.Label(master, text="Randles w/ Warburg", font=font2).grid(row=2,column=1)
        
        pseudo_btn = tkinter.Button(master, image = self.psuedo_pic, command = self.psuedo_calc_impedance)
        pseudo_btn.image = self.psuedo_pic
        pseudo_btn.grid(row=3,column=0)
        tkinter.Label(master, text="Psuedocapacitance", font=font2).grid(row=4,column=0)

        five_trans_btn = tkinter.Button(master, image = self.five_transmission_pic, command = self.five_transmission_calc_impedance)
        five_trans_btn.image = self.five_transmission_pic
        five_trans_btn.grid(row=3,column=1)
        tkinter.Label(master, text="Five Term Transmission", font=font2).grid(row=4,column=1)
        
    def srand_calc_impedance(self):
        self.data_Z, self.data_freq = file_select()
        circuit = 'R0-p(CPE0,R1)'
        initial_guess = [20,.1,.1,50]
        circuit = CustomCircuit(circuit, initial_guess=initial_guess)
        circuit.fit(self.data_freq, self.data_Z)
        Z_fit = circuit.predict(self.data_freq)

        fig, ax = plt.subplots()
        plot_nyquist(ax,self.data_Z, fmt="o")
        plot_nyquist(ax, Z_fit, fmt="-")
        plt.legend(["Data", "Fit"])

        self.master.withdraw()
        self.eqc_calc_menu = tkinter.Toplevel()

        img_canvas = tkinter.Canvas(self.eqc_calc_menu, width=self.s_rand_pic.width(), height=self.s_rand_pic.height())
        img_canvas.grid(row=0, column =0)
        img_canvas.create_image(0,0, image=self.s_rand_pic, anchor="nw")
        img_canvas.update()

        self.graph_canvas = FigureCanvasTkAgg(fig, self.eqc_calc_menu)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().grid(row=0, rowspan = 3, column=1)

        return_btn = tkinter.Button(self.eqc_calc_menu, text="Return to Main Menu", font=("Helvetica", 14), bg="OliveDrab1", command = lambda:[self.eqc_calc_menu.destroy(), self.master.deiconify()])
        return_btn.grid(row = 3, column = 0)

        exit_btn = tkinter.Button(self.eqc_calc_menu, text="Exit Program", font=("Helvetica", 14), bg="tomato", command = lambda:[self.eqc_calc_menu.destroy(), self.master.destroy()])
        exit_btn.grid(row = 4, column = 0)

        stats = tkinter.Label(self.eqc_calc_menu, text =circuit, font=("Helvetica", 12), justify="left")
        stats.grid(row = 2, column = 0)
        
    def warburg_calc_impedance(self):
        self.data_Z, self.data_freq = file_select()
        circuit = 'R0-p(CPE0,R1-W0)'
        initial_guess = [20,.1,.1,50, 50]
        circuit = CustomCircuit(circuit, initial_guess=initial_guess)
        circuit.fit(self.data_freq, self.data_Z)
        Z_fit = circuit.predict(self.data_freq)

        fig, ax = plt.subplots()
        plot_nyquist(ax,self.data_Z, fmt="o")
        plot_nyquist(ax, Z_fit, fmt="-")
        plt.legend(["Data", "Fit"])

        self.master.withdraw()
        self.eqc_calc_menu = tkinter.Toplevel()

        img_canvas = tkinter.Canvas(self.eqc_calc_menu, width=self.warburg_pic.width(), height=self.warburg_pic.height())
        img_canvas.grid(row=0, column =0)
        img_canvas.create_image(0,0, image=self.warburg_pic, anchor="nw")
        img_canvas.update()

        self.graph_canvas = FigureCanvasTkAgg(fig, self.eqc_calc_menu)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().grid(row=0, rowspan = 3, column=1)

        return_btn = tkinter.Button(self.eqc_calc_menu, text="Return to Main Menu", font=("Helvetica", 14), bg="OliveDrab1", command = lambda:[self.eqc_calc_menu.destroy(), self.master.deiconify()])
        return_btn.grid(row = 3, column = 0)

        exit_btn = tkinter.Button(self.eqc_calc_menu, text="Exit Program", font=("Helvetica", 14), bg="tomato", command = lambda:[self.eqc_calc_menu.destroy(), self.master.destroy()])
        exit_btn.grid(row = 4, column = 0)

        stats = tkinter.Label(self.eqc_calc_menu, text =circuit, font=("Helvetica", 12), justify="left")
        stats.grid(row = 2, column = 0)
    
    def psuedo_calc_impedance(self):
        self.data_Z, self.data_freq = file_select()
        circuit = 'R0-p(CPE0,R1-p(CPE1,R2))'
        initial_guess = [20,.1,.1,50, 50,.1,50]
        circuit = CustomCircuit(circuit, initial_guess=initial_guess)
        circuit.fit(self.data_freq, self.data_Z)
        Z_fit = circuit.predict(self.data_freq)

        fig, ax = plt.subplots()
        plot_nyquist(ax,self.data_Z, fmt="o")
        plot_nyquist(ax, Z_fit, fmt="-")
        plt.legend(["Data", "Fit"])

        self.master.withdraw()
        self.eqc_calc_menu = tkinter.Toplevel()

        img_canvas = tkinter.Canvas(self.eqc_calc_menu, width=self.psuedo_pic.width(), height=self.psuedo_pic.height())
        img_canvas.grid(row=0, column =0)
        img_canvas.create_image(0,0, image=self.psuedo_pic, anchor="nw")
        img_canvas.update()

        self.graph_canvas = FigureCanvasTkAgg(fig, self.eqc_calc_menu)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().grid(row=0, rowspan = 3, column=1)

        return_btn = tkinter.Button(self.eqc_calc_menu, text="Return to Main Menu", font=("Helvetica", 14), bg="OliveDrab1", command = lambda:[self.eqc_calc_menu.destroy(), self.master.deiconify()])
        return_btn.grid(row = 3, column = 0)

        exit_btn = tkinter.Button(self.eqc_calc_menu, text="Exit Program", font=("Helvetica", 14), bg="tomato", command = lambda:[self.eqc_calc_menu.destroy(), self.master.destroy()])
        exit_btn.grid(row = 4, column = 0)

        stats = tkinter.Label(self.eqc_calc_menu, text =circuit, font=("Helvetica", 12), justify="left")
        stats.grid(row = 2, column = 0)

    def five_transmission_calc_impedance(self):
        self.data_Z, self.data_freq = file_select()
        circuit = 'R0-p(CPE0,R1-p(CPE1,R2))'
        initial_guess = [20,.1,.1,50, 50,.1,50]
        circuit = CustomCircuit(circuit, initial_guess=initial_guess)
        circuit.fit(self.data_freq, self.data_Z)
        Z_fit = circuit.predict(self.data_freq)

        fig, ax = plt.subplots()
        plot_nyquist(ax,self.data_Z, fmt="o")
        plot_nyquist(ax, Z_fit, fmt="-")
        plt.legend(["Data", "Fit"])

        self.master.withdraw()
        self.eqc_calc_menu = tkinter.Toplevel()

        img_canvas = tkinter.Canvas(self.eqc_calc_menu, width=self.five_transmission_pic.width(), height=self.five_transmission_pic.height())
        img_canvas.grid(row=0, column =0)
        img_canvas.create_image(0,0, image=self.five_transmission_pic, anchor="nw")
        img_canvas.update()

        self.graph_canvas = FigureCanvasTkAgg(fig, self.eqc_calc_menu)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().grid(row=0, rowspan = 3, column=1)

        return_btn = tkinter.Button(self.eqc_calc_menu, text="Return to Main Menu", font=("Helvetica", 14), bg="OliveDrab1", command = lambda:[self.eqc_calc_menu.destroy(), self.master.deiconify()])
        return_btn.grid(row = 3, column = 0)

        exit_btn = tkinter.Button(self.eqc_calc_menu, text="Exit Program", font=("Helvetica", 14), bg="tomato", command = lambda:[self.eqc_calc_menu.destroy(), self.master.destroy()])
        exit_btn.grid(row = 4, column = 0)

        stats = tkinter.Label(self.eqc_calc_menu, text =circuit, font=("Helvetica", 12), justify="left")
        stats.grid(row = 2, column = 0)

    def go_back(self):
        self.eqc_calc_menu.destroy()
        self.master.deiconify()

#root = tkinter.Tk()
#start = MainWindow(root)
#root.mainloop()


def lmfit_method():
    ## var = [R0, CPE0_0, CPE0_1, R1]
    R0, CPE0_0, CPE0_1, R1 = [5.83, 2.37e-03, 5.68e-01, 8.73e+02]
    frequency = [1e+05, 1e-02]
    x = np.logspace(float(-2),float(5), num=200)*2*np.pi
    

    #Real data
    randlesmodel = SimpleRandlesModel(simple_randles_eq)
    data_Z, data_freq = file_select()
    trueZ = data_Z
    guess = randlesmodel.guess(trueZ, x=data_freq, verbose=True)
    result = randlesmodel.fit(trueZ, params=guess, x=data_freq, verbose=True)
    guess_eval = randlesmodel.eval(params=guess, x=data_freq)
    result_eval = randlesmodel.eval(params=result.params, x=data_freq)


    #TEST DATA
    #randlesmodel = SimpleRandlesModel(simple_randles_eq)
    #true_params = randlesmodel.make_params(x = x, R0 = R0, CPE0_0 = CPE0_0, CPE0_1 = CPE0_1, R1 = R1)
    #true_data = randlesmodel.eval(params=true_params, x=x)
    #trueZ = simple_randles_eq(x, R0, CPE0_0, CPE0_1, R1)
    #guess = randlesmodel.guess(true_data, x=x, verbose=True)
    #result = randlesmodel.fit(true_data, params=guess, x=x, verbose=True)
    #guess_eval = randlesmodel.eval(params=guess, x=x)
    #result_eval = randlesmodel.eval(params=result.params, x=x)

    print(result.fit_report() + "\n")
    result.params.pretty_print()

    plt.figure()
    plot_ri(trueZ, ".", label="Data")
    #plot_ri(guess_eval, "k--", label="Guess Fit")
    plot_ri(result_eval, "r-", label = "Best Fit")
    plt.legend(loc="best")
    plt.xlabel("Re")
    plt.ylabel("Im")

    plt.show()

def lmfit_method2():
    Rs, Rm, L, Cw, Rw, Cu, Ru = [200e03, 12e6, 0.025, 700e-6, 30e3, 0.6e-6, 280e3]
    freq = np.logspace(float(-3),float(1), num=200)*2*np.pi
    print(freq)

    

    #Real data
    #randlesmodel = SimpleRandlesModel(simple_randles_eq)
    #data_Z, data_freq = file_select()
    #trueZ = data_Z
    #guess = randlesmodel.guess(trueZ, x=data_freq, verbose=True)
    #result = randlesmodel.fit(trueZ, params=guess, x=data_freq, verbose=True)
    #guess_eval = randlesmodel.eval(params=guess, x=data_freq)
    #result_eval = randlesmodel.eval(params=result.params, x=data_freq)


    #TEST DATA
    model = CylindricalModel(cylindrical_pore)
    true_params = model.make_params(freq = freq,  Rs = Rs, Rm = Rm, L = L, Cw = Cw, Rw = Rw, Cu = Cu, Ru = Ru)
    true_data = model.eval(params=true_params, freq=freq)
    trueZ = cylindrical_pore(freq, Rs, Rm, L, Cw, Rw, Cu, Ru)
    guess = model.guess(true_data, freq=freq, verbose=True)
    #result = model.fit(true_data, params=guess, freq=freq, verbose=True)
    #guess_eval = model.eval(params=guess, freq=freq)
    #result_eval = model.eval(params=result.params, freq=freq)

    print(trueZ)
    #print(result.fit_report() + "\n")
    #result.params.pretty_print()

    plt.figure()
    plot_ri(trueZ, "b-", label="Data")
    #plot_ri(guess_eval, "k--", label="Guess Fit")
    #plot_ri(result_eval, "r-", label = "Best Fit")
    plt.legend(loc="best")
    plt.xlabel("Re")
    plt.ylabel("Im")

    plt.show()

lmfit_method2()