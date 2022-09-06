import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from statistics import mean

# uses fit_and_lumidata_averageBCIDs.csv
# makes stability plot of sigvis vs scan (sigmavis_vs_scan)
# make same plot of sigvis vs scan but normalised by average sigvis (sigmavis_vs_scan_scaled).
# But better to use makeplots comparison.py if want to compare different stability plots

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('output', nargs='?', help='Output directory to store plots.')
    #args = parser.parse_args()

    output_dir = '/Users/leila/PycharmProjects/DESY/VDM' #args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dets = ["PLT"] # "BCM1F", "HFOC", "HFET", "PLT"]
    fits = ["SG"] # "SGConst", "DG", "SG","twoMuDG"]
    corrs = ["Background"] #, "noCorr"
    energies = ["13600"]

    wantSigVisVsScan = True # super stability plot
    wantSigVisVsScan_scaled = False # same stability plot but scaled so can compare to other detectors
    wantSigVisVsSBIL = False # not a very good plot
    wantSigVisVsFill = False # not a very good plot

    #fills_13TeV = [7679, 7725, 7816, 7886] #8027 is one from july 22
    #fills_13TeV = [7679, 7725, 7816, 7886, 8007]
    fills_13TeV = [7966,7978,8007, 8016, 8027,8058,8063,8067,8068,8072,8078,8079,8081,8087,8088,8091,8094,8102,8103,8106,8112,8113,8115,8118,8120,8124,8128,8132,8136,8142,8143,8144,8146,8147,8148,8149,8151] #7920, 7921,7960,7963,
    blacklist_fill = []

    df = pd.read_csv("fit_and_lumidata_averageBCIDs.csv", sep=',') #this file contains averages over the BCIDs for each scan.
    sigma_vis = df["avg_xsec"]

    for det in dets:
        for corr in corrs:
            for fit in fits:
                for energy in energies:
                    print("---------------------------")
                    print("ENERGY: ", energy)
                    print("DET: ", det)
                    print("CORR: ", corr)
                    print("FITS: ", fit)
                    partial_df = df.loc[(df["luminometer"] == det) & (df["fit"] == fit) & (df["correction"] == corr)]

                    allfills = list(np.asarray(partial_df['fill']))
                    fills = list(dict.fromkeys(allfills)) #this deletes all repeated fills so we have a list of all unique fill numbers

                    # Remove fills at 13 TeV
                    fills = [x for x in fills if not x in fills_13TeV]
                    fills = [x for x in fills if not x in blacklist_fill]
                    fills_13TeV = [x for x in fills_13TeV if not x in blacklist_fill]
                    fills = fills if energy == "900" else fills_13TeV
                    print("Fills: ", fills)

                    sigma_vis = []
                    sbil = [] #single bunch instantaneous luminosity
                    sigma_vis_err = []
                    sbil_err = []
                    eff_fills = []
                    fillcounter = [] #for the plot of sigma vis as a fkt of scan, to know where to draw the vertical lines separating diff fills
                    for fill in fills:
                        if "HF" in det and fill == 7733:
                            continue
                        df_samefill = partial_df[partial_df["fill"] == fill]
                        if np.isnan(np.mean(df_samefill["avg_xsec"])): # or np.isnan(np.sqrt(df_samefill["SBILErr"].pow(2).sum())):
                            continue

                        fillcounter.append([len(sigma_vis),fill])  # the length of sigma vis should give me the index the fill starts at, and what fill it is
                        print(len(sigma_vis),fill)

                        sigma_vis += df_samefill["avg_xsec"].tolist()
                        sbil += df_samefill["avg_SBIL"].tolist()
                        sigma_vis_err += df_samefill["avg_xsecErr"].tolist()
                        #sbil_err += df_samefill["SBILErr"].tolist()

                        for i in range(len(df_samefill["avg_xsec"].tolist())):
                            #eff_fills.append(fill-i*2)
                            eff_fills.append(fill+0.1*i) #should be pos?




                    print("Effective fills: ", eff_fills)
                    print("SBIL: ", sbil)
                    print("sigmavis: ", sigma_vis)
                    if len(sigma_vis) > 0:
                        energy_str = "13.6 TeV" if energy == "13600" else "900 GeV"
                        ## Plot sigma_vis vs. SBIL
                        if wantSigVisVsSBIL:
                            f = plt.figure()
                            ax = f.add_subplot(111) #1 by 1 grid of subplots, first subplot.
                            plt.scatter(sbil, sigma_vis, marker=".", alpha = 1)
                            plt.errorbar(sbil, sigma_vis, yerr=sigma_vis_err, fmt="o", alpha = 1)
                            plt.xlabel('SBIL', size=13)
                            plt.ylabel(r'$\sigma_{vis}$', size=13)
                            plt.xlim([0, 6])
                            if any(val > 5 for val in sbil): plt.xlim([0, max(sbil) + 1])
                            if energy == "13600":
                                if det == "BCM1F":
                                      plt.ylim([100, 180])
                                elif det == "PLT":
                                      plt.ylim([250, 370])
                                elif det == "HFOC":
                                      plt.ylim([600, 1150])
                                elif det == "HFET":
                                      plt.ylim([1600, 3600])
                            plt.text(0.06, 1.04,'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.28, 1.04,'Work in progress', ha='center', va='center', transform=ax.transAxes, style='italic', size=13)
                            plt.text(0.87, 1.04,f'{energy_str}', ha='center', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.03, 0.95,f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.89,f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.83,f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.77,f'Average over BCIDs for each scan', ha='left', va='center', weight = 'bold', transform=ax.transAxes, size=13)
                            plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_{det}_{fit}_{corr}_{energy}.pdf")
                            plt.clf()
                            del f

                        ## Plot sigma_vis vs. fill
                        if wantSigVisVsFill:
                            f = plt.figure()
                            ax = f.add_subplot(111)
                            plt.scatter(eff_fills, sigma_vis, marker=".", alpha = 1)
                            plt.errorbar(eff_fills, sigma_vis, yerr=sigma_vis_err, fmt="o", alpha= 1)
                            plt.xlabel('Fill', size=13)
                            plt.ylabel(r'$\sigma_{vis}$', size=13)
                            plt.text(0.06, 1.04,'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.28, 1.04,'Work in progress', ha='center', va='center', transform=ax.transAxes, style='italic', size=13)
                            plt.text(0.87, 1.04,f'{energy_str}', ha='center', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.03, 0.95,f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.89,f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.83,f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.77, f'Average over BCIDs for each scan', ha='left', va='center', weight='bold', transform=ax.transAxes, size=13)
                            plt.xlim([np.min(eff_fills)-50, np.max(eff_fills)+50])
                            if energy == "13600":
                                if det == "BCM1F":
                                      plt.ylim([100, 180])
                                elif det == "PLT":
                                      plt.ylim([250, 370])
                                elif det == "HFOC":
                                      plt.ylim([600, 1150])
                                elif det == "HFET":
                                      plt.ylim([1600, 3600])
                            plt.savefig(f"{output_dir}/sigmavis_vs_fill_{det}_{fit}_{corr}_{energy}.pdf")
                            plt.clf()
                            del f

                        if wantSigVisVsScan:
                            #plot of sigma vis as a function of scan number, so that they are equidistant
                            num_scans = len(eff_fills)
                            enumerate_scans = list(range(num_scans))

                            f = plt.figure(figsize=(9,5.5))
                            ax = f.add_subplot(111)
                            plt.vlines(x=[x[0] - 0.5 for x in fillcounter], ymin=0, ymax=4000, colors=['purple'], ls='--', lw=1, alpha = 0.5)
                            avg_sigma_vis = mean(sigma_vis)
                            x = np.arange(0, num_scans, 1)
                            avg_sigma_vis_array = np.full_like(x, avg_sigma_vis, dtype=np.double)
                            plt.plot(x,avg_sigma_vis_array,label=f'avg={round(avg_sigma_vis,2)}'r'$\mu b$', color = 'darkorange',zorder=1)
                            def rmse(theoretical, data):
                                return np.sqrt(mean((theoretical - data) ** 2))
                            plt.fill_between(x, avg_sigma_vis_array+rmse(avg_sigma_vis_array,sigma_vis) , avg_sigma_vis_array-rmse(avg_sigma_vis_array,sigma_vis),
                                             alpha = 0.5, label=f'rms={round(rmse(avg_sigma_vis_array,sigma_vis),2)}'r'$\mu b$', color = 'darkorange', zorder=2)
                            plt.errorbar(enumerate_scans, sigma_vis, yerr=sigma_vis_err, alpha=0.6, ls="none", color="gray", zorder = 3)

                            plt.scatter(enumerate_scans, sigma_vis, c=sbil, cmap='viridis', vmin=0, vmax=max(sbil), marker=".", alpha=1, s=100, zorder = 4)
                            cbar = plt.colorbar()
                            cbar.set_label(r'SBIL [$Hz/\mu b$]', rotation=270,labelpad=+15,size='large')
                            plt.legend()
                            plt.xlabel('Scan', size=15)
                            plt.ylabel(r'$\sigma_{vis} \,[\mu b]$', size=15)

                            plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.22, 1.04, 'Preliminary', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
                            plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.77, f'Average over BCIDs for each scan', ha='left', va='center', weight='bold',transform=ax.transAxes, size=13)

                            '''
                            #official format for joanna
                            plt.text(0.03, 0.95, 'CMS', ha='left', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.13, 0.95, 'Preliminary', ha='left', va='center', transform=ax.transAxes, style='italic', size=13)
                            plt.text(0.87, 1.04, f'(2022, {energy_str})', ha='center', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.89, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.83, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.77, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.71, f'Average over BCIDs for each scan', ha='left', va='center', weight='bold', transform=ax.transAxes, size=13)
                            '''

                            plt.xlim([np.min(enumerate_scans) - 1, np.max(enumerate_scans) + 1])
                            if energy == "13600":
                                if det == "BCM1F":
                                    plt.ylim([100, 180])
                                elif det == "PLT":
                                    plt.ylim([240, 370])
                                elif det == "HFOC":
                                    plt.ylim([600, 1200])
                                elif det == "HFET":
                                    plt.ylim([2400, 4000])
                            i=0
                            height = 120
                            if det == "HFOC": height= 700
                            if det == "HFET": height = 2700
                            if det == "PLT": height = 260
                            for x in fillcounter:
                                xpos = x[0]-0.4
                                plt.text(xpos, height, f'FILL {fillcounter[i][1]}', horizontalalignment='left', verticalalignment='center',rotation='vertical',size=8,zorder=7)
                                i+=1
                            plt.savefig(f"{output_dir}/sigmavis_vs_scan_{det}_{fit}_{corr}_{energy}.pdf")
                            plt.clf()
                            del f

                            if wantSigVisVsScan_scaled:
                                #plot of sigvis vs scan but normalised, to compare for diff luminometers/fits
                                f = plt.figure(figsize=(8, 5))
                                ax = f.add_subplot(111)
                                plt.vlines(x=[x[0] - 0.5 for x in fillcounter], ymin=0, ymax=4000, colors=['purple'],
                                           ls='--', lw=1, alpha=0.4)
                                avg_sigma_vis = mean(sigma_vis)
                                x = np.arange(0, num_scans, 1)
                                #plt.fill_between(x, avg_sigma_vis_array + rmse(avg_sigma_vis_array, sigma_vis),
                                #                 avg_sigma_vis_array - rmse(avg_sigma_vis_array, sigma_vis),
                                #                 alpha=0.5, label=f'rmse ={round(rmse(avg_sigma_vis_array, sigma_vis), 2)}',
                                #                 color='darkorange', zorder=2)

                                sigma_vis_scaled = [val/avg_sigma_vis for val in sigma_vis]
                                sigma_vis_err_scaled = [val/avg_sigma_vis for val in sigma_vis_err]
                                plt.scatter(enumerate_scans, sigma_vis_scaled, c=sbil, cmap='viridis', vmin=0, vmax=max(sbil),marker=".", alpha=1, s=100, zorder=4)
                                plt.errorbar(enumerate_scans, sigma_vis_scaled, yerr=sigma_vis_err_scaled, alpha=1, ls="none",color="gray", zorder=3)
                                avg_sigma_vis_array = np.full_like(x, 1, dtype=np.double)
                                plt.plot(x, avg_sigma_vis_array, label=r'$\overline{\sigma_{vis}}$'f'={round(avg_sigma_vis, 2)}', color='darkorange', zorder=1) #plotting the line of average sigvis

                                cbar = plt.colorbar()
                                cbar.set_label('SBIL', rotation=270, labelpad=+15)
                                plt.legend()
                                plt.xlabel('Scan', size=15)
                                plt.ylabel(r'$\sigma_{vis} \, / \, \overline{\sigma_{vis}}$', size=15)

                                plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
                                plt.text(0.28, 1.04, 'Work in progress', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
                                plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes,weight="bold", size=14)
                                plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.text(0.03, 0.77, f'Average over BCIDs for each scan', ha='left', va='center',weight='bold', transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.71, f'Sigma visible scaled using average', ha='left', va='center',transform=ax.transAxes, size=13)

                                plt.xlim([np.min(enumerate_scans) - 1, np.max(enumerate_scans) + 1])
                                plt.ylim([0.7,1.3])

                                '''if energy == "13600":
                                    if det == "BCM1F":
                                        plt.ylim([100/avg_sigma_vis, 180/avg_sigma_vis])
                                    elif det == "PLT":
                                        plt.ylim([250/avg_sigma_vis, 370/avg_sigma_vis])
                                    elif det == "HFOC":
                                        plt.ylim([600/avg_sigma_vis, 1200/avg_sigma_vis])
                                    elif det == "HFET":
                                        plt.ylim([2400/avg_sigma_vis, 4000/avg_sigma_vis])'''
                                i = 0
                                height = 0.8
                                #if det == "HFOC": height = 700
                                #if det == "HFET": height = 2700
                                for x in fillcounter:
                                    xpos = x[0] - 0.5
                                    plt.text(xpos, height, f'FILL {fillcounter[i][1]}', horizontalalignment='left', verticalalignment='center', rotation='vertical')
                                    i += 1
                                plt.savefig(f"{output_dir}/sigmavis_vs_scan_scaled_{det}_{fit}_{corr}_{energy}.pdf")
                                plt.clf()
                                del f

                        '''
                        #calculating the correlation coeff
                        cov_matrix = np.cov(sbil, sigma_vis)
                        covariance = cov_matrix[0][1]
                        variance_sbil = cov_matrix[0][0]
                        variance_sigma_vis = cov_matrix[1][1]
                        correl_coeff = covariance/np.sqrt(variance_sbil*variance_sigma_vis)
                        print(correl_coeff)
                        '''