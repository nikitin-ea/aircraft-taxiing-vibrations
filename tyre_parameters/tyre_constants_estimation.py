import tyreLoadCurveFitting as fit
import tyreLoadCurvePlotting as plot

##############################################################################
patch_form = {'ellipse' : 1, 'square' : 0.5}

def model(xdata, c1, c2):
    form = 'square'
    deflection = xdata[:,0]
    pressure = xdata[:,1]
    return (deflection**(1 + patch_form[form]) / 
            (c1 + c2 * deflection**(patch_form[form]) / pressure))

##############################################################################
PATH = r"C:\Users\devoi\Thesis\MLG\MLG\MLG_tyre"
bounds = ((0.00001, 0.00001), (0.01, 0.01))
fit.MAX_DEFL = 210.0
plot.GRID_PTS = 100

if __name__ == "__main__":
    raw_data = fit.load_data(PATH)
    xdata, ydata, curves, pressures = fit.prepare_data(raw_data)
    popt = fit.fit_model(model, xdata, ydata, bounds)
    if popt is not None:
        plot.plot_all(xdata, ydata, raw_data, model, popt)
