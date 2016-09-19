import sys
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__ == '__main__':
    if __package__ is None:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.data_handler import get_system
    else:
        from scripts.data_handler import get_system


def colored_errorbar(axis, system, x=None, y=None, xerr=None, yerr=None):
    sc = axis.scatter(x, y, s=20, c=system.time, edgecolors='none',
                      cmap=plt.get_cmap("viridis_r"))
    
    clb = plt.colorbar(sc) # create a colorbar from the scatter colors
    clb.remove() # but don't show it

    a,b,c = axis.errorbar(x, y, xerr=xerr, yerr=yerr,
                          fmt='o', ms=0, zorder=0, mew=0)

    # convert time to a color tuple using the colormap used for scatter
    time_color = clb.to_rgba(system.time)
    # adjust the color of the errorbars to the colormap
    try:
        c[0].set_color(time_color)
        c[1].set_color(time_color)
    except IndexError:
        pass



def make_correlation_plot(system):
    fig = plt.figure('Planet System %d' % system.number,
                     figsize=(12,10))

    gs = gridspec.GridSpec(3, 3)

    # gs.update(left=0.55, right=0.98, hspace=0.05)

    ### RV vs time
    ##############
    ax1 = plt.subplot(gs[:-2, :-1])
    colored_errorbar(ax1, system, x=system.time, y=system.vrad, yerr=system.error)
    ax1.set_xlabel('BJD - 2400000 [days]')
    ax1.set_ylabel('RV [m/s]')

    ### RV vs BIS
    #############
    ax2 = plt.subplot(gs[0, -1])
    colored_errorbar(ax2, system, x=system.extras.bis_span*1e3, y=system.vrad)
    ax2.set_xlabel('BIS')
    ax2.set_ylabel('RV')

    ### Rhk vs RV
    #############
    ax3 = plt.subplot(gs[1, 1])
    colored_errorbar(ax3, system, x=system.vrad, y=system.extras.rhk,
                     xerr=system.error, yerr=system.extras.sig_rhk)
    ax3.set_xlabel('RV [m/s]')
    ax3.set_ylabel("R'hk")

    ### Rhk vs BIS
    ##############
    ax4 = plt.subplot(gs[1, 2])
    colored_errorbar(ax4, system, x=system.extras.bis_span*1e3, y=system.extras.rhk,
                     yerr=system.extras.sig_rhk)
    ax4.set_xlabel('BIS')
    ax4.set_ylabel("R'hk")


    ### FWHM vs Rhk
    ###############
    ax5 = plt.subplot(gs[2, 0])
    colored_errorbar(ax5, system, x=system.extras.rhk, y=system.extras.fwhm*1e3,
                     xerr=system.extras.sig_rhk)
    ax5.set_xlabel("R'hk")
    ax5.set_ylabel("FWHM")


    ### FWHM vs RV
    ##############
    ax6 = plt.subplot(gs[2, 1])
    colored_errorbar(ax6, system, x=system.vrad, y=system.extras.fwhm*1e3,
                     xerr=system.error)
    ax6.set_xlabel('RV [m/s]')
    ax6.set_ylabel("FWHM")


    ### FWHM vs BIS
    ###############
    ax7 = plt.subplot(gs[2, 2])
    colored_errorbar(ax7, system, x=system.extras.bis_span*1e3, y=system.extras.fwhm*1e3)
    ax7.set_xlabel('BIS')
    ax7.set_ylabel("FWHM")


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        i = int(sys.argv[1])
        system = get_system(number=i)
        make_correlation_plot(system)
    except IndexError, e:
        print __file__, 'system_number'
# system_numbers = [i for i in range(1, 16) if i!=6]
# for i in system_numbers:
#     system = get_system(number=i)
#     make_correlation_plot(system)