import pandas as pd
import numpy as np
import shutil
import os
import argparse
import matplotlib.pyplot as plt
from statistics import mean

# makes plot of sigvis vs SBIL, points color coded according to their BCID (sigmavis_vs_SBIL_color_coded_bcid)
# makes plot of sigvis vs BCID for a given scan (sigmavis_vs_BCID)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('output', nargs='?', help='Output directory to store plots.')
    #args = parser.parse_args()

    output_dir = '/Users/leila/PycharmProjects/DESY/VDM' #args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dets = ["BCM1F"] # "HFOC", "HFET", "PLT", "BCM1F"]
    fits = ["twoMuDG"] # "SGConst", "DG", "SG","twoMuDG"]
    corrs = ["Background"] # "Background", "noCorr"
    energies = ["13600"]

    #fills_13TeV = [8007, 8016, 8027,8058,8063,8067,8068,8072,8078,8079,8081]
    fills_13TeV = [8088]
    blacklist_fill = []

    wantSigVisVsBCID = False
    wantSigVisVsSBIL_ColorCodedBCID = True
    is_single_fill = True

    df = pd.read_csv("fit_and_lumidata.csv", sep=',')

    sigma_vis = df["xsec"]


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

                    if len(fills) > 1: is_single_fill = False

                    sigma_vis = []
                    sbil = [] #single bunch instantaneous luminosity
                    sigma_vis_err = []
                    sbil_err = []
                    eff_fills = []
                    bcids = []
                    badfills = []
                    sbils_not_avgd = [] #to be a list of lists, one inner list containing all the SBILs in one scan
                    sigma_vis_not_avgd = []
                    sigma_vis_err_not_avgd = []
                    for fill in fills:
                        if "HF" in det and fill == 7733:
                            continue
                        df_samefill = partial_df[partial_df["fill"] == fill]
                        #print(df_samefill)
                        if np.isnan(np.mean(df_samefill["xsec"])) or np.isnan(np.sqrt(df_samefill["SBILErr"].pow(2).sum())):
                            print(fill, ":(")
                            badfills.append(fill)
                            continue

                        allscans_in_fill = list(np.asarray(df_samefill['scan_name'])) #gets all the scan names of the files of the same fill number
                        scans_in_fill = list(dict.fromkeys(allscans_in_fill)) #makes a list of the unique scan names from that. So if there were 2 scans in the fill, we would have 2 scan names
                        if len(scans_in_fill) ==1: is_single_scan = True
                        else: is_single_scan = False

                        for scan in scans_in_fill:
                            print("\n SCAN:", scan, "OF TOTAL SCANS IN FILL:", scans_in_fill)
                            df_samescan = partial_df[partial_df["scan_name"] == scan] #making a dataframe of things from the same scan
                            avg_sigmavis_of_scan = np.mean(df_samescan["xsec"])
                            sigma_vis.append(avg_sigmavis_of_scan)
                            print("FILL:", fill, "ADDED TO SIG VIS:", avg_sigmavis_of_scan)
                            sbil.append(np.mean(df_samescan["SBIL"]))
                            sigma_vis_err.append(np.sqrt(df_samescan["xsecErr"].pow(2).sum()/(df_samescan.shape[0]))) #dividing by number of entries averaged. shape[0] of a df returns number of rows
                            sbil_err.append(np.sqrt(df_samescan["SBILErr"].pow(2).sum()))

                            #for plot of sigvis vs sbil where I need no averages, but separate lists for separate scans
                            sbils_current_scan = list(np.asarray(df_samescan["SBIL"]))
                            sbils_not_avgd.append(sbils_current_scan)
                            sigma_vis_current_scan = list(np.asarray(df_samescan["xsec"]))
                            sigma_vis_not_avgd.append(sigma_vis_current_scan)
                            sigma_vis_err_current_scan = list(np.asarray(df_samescan["xsecErr"]))
                            sigma_vis_err_not_avgd.append(sigma_vis_err_current_scan)
                            bcids_current_scan = list(np.asarray(df_samescan["BCID"]))
                            bcids.append(bcids_current_scan)

                        for i in range(len(scans_in_fill)):
                            eff_fills.append(fill+i*2)
                            #eff_fills.append(fill-0.1*i)
                        print("*********************** eff fills:", eff_fills)
                        print("sigma vis:", sigma_vis)
                        print("sigma vis errors:", sigma_vis_err)
                        #bcids += df_samefill["BCID"].tolist()
                    #fills = [x for x in fills if x not in badfills] #get rid of badfills

                    print("\n\nEffective fills: ", eff_fills)
                    print("SBIL: ", sbil)
                    print("sigmavis: ", sigma_vis)
                    print("BCIDs: ", bcids)

                    #making plots
                    #sigvis vs SBIL color coded by scan
                    if is_single_fill:  # and len(sbils_not_avgd)>1:
                        if wantSigVisVsSBIL_ColorCodedBCID:
                            for i in range(len(sbils_not_avgd)):  # looping over each scan to remove extreme outliers
                                # To get rid of the huge outliers.(see makeplots current version). Here want to get BCID of these outliers to investigate
                                # Necessary for twoMuDG 8007,8030,8033,8057,8078,8079,8081,8088,8103,8106.
                                point = 0
                                huge_points = []
                                while point < len(sigma_vis_not_avgd[i]) - 1:  # looping over the index of all the elements in the list of sigma vis for one scan.
                                    if sigma_vis_not_avgd[i][point] > 500 or sbils_not_avgd[i][point] > 20:  # 6000
                                        huge_points.append(bcids[i].pop(point))
                                        huge_points.append(sigma_vis_not_avgd[i].pop(point))
                                        sbils_not_avgd[i].pop(point)
                                    else:
                                        point += 1
                                print("HUGE POINTS REMOVED (bcid followed by its sigmavis):", huge_points)
                            for i in range(len(sbils_not_avgd)):  # looping over each scan now to plot
                                f = plt.figure(figsize=(8, 5))
                                ax = f.add_subplot(111)  # 1 by 1 grid of subplots, first subplot

                                # linear fit with avg SBIL in order to decorrelate the 2 fit parameters
                                avg_sbil_of_scan = mean(sbils_not_avgd[i])
                                linfit, cov = np.polyfit(sbils_not_avgd[i] - avg_sbil_of_scan, sigma_vis_not_avgd[i], 1, cov=True)
                                err = np.sqrt(np.diag(cov))
                                plt.plot(sbils_not_avgd[i],[element * linfit[0] - avg_sbil_of_scan * linfit[0] + linfit[1] for element in sbils_not_avgd[i]],color='black', label=f'({round(linfit[0], 2)}$\pm${round(err[0], 2)})(x-avgSBIL) + {round(linfit[1], 2)}$\pm${round(err[1], 2)}')  # - avg_sbil_of_scan* linfit[0] is to make line correct w pts

                                # plotting the points in each scan
                                plt.scatter(sbils_not_avgd[i], sigma_vis_not_avgd[i], c=bcids[i], cmap='viridis', vmin=0, vmax=max(bcids[i]),marker=".", alpha=1, s=10)  # for fit with decorrelated params
                                # plt.errorbar(sbils_not_avgd[i],sigma_vis_not_avgd[i], yerr=sigma_vis_err_not_avgd[i], alpha=1, ls='none', color = colors[i]) #errors are too small to make sense to plot

                                cbar = plt.colorbar()
                                cbar.set_label('BCID', rotation=270, labelpad=+15)

                                '''
                                #TEST getting the bcids in different clusters
                                #x = np.arange(0, 10, 1)
                                #test_array = np.full_like(x, 150, dtype=np.double)
                                #plt.plot(x, test_array, color='darkorange',zorder=1)
                                bcids_in_top_cluster = []
                                bcids_in_bottom_cluster = []
                                for point in range(len(bcids[i])):
                                    if sigma_vis_not_avgd[i][point] > 150: bcids_in_top_cluster.append(bcids[i][point])
                                    if sigma_vis_not_avgd[i][point] < 150: bcids_in_bottom_cluster.append(bcids[i][point])
                                print("TOP CLUSTER:", bcids_in_top_cluster)
                                print("BOTTOM CLUSTER:", bcids_in_bottom_cluster)
                                # TEST over
                                '''

                                plt.xlabel('SBIL', size=13)
                                plt.ylabel(r'$\sigma_{vis}$', size=13)
                                plt.legend(fontsize=7.5)  # loc = 3 is bottom left
                                # adjusting the xlim
                                maximumx = 0
                                minimumx = 3
                                for l in sbils_not_avgd:
                                    if max(l) > maximumx: maximumx = max(l)
                                    if min(l) < minimumx: minimumx = min(l)
                                plt.xlim([minimumx - 0.5, maximumx + 0.5])
                                if energy == "13600":
                                    if det == "BCM1F":
                                        minimumy, maximumy = 140, 140
                                        for l in sigma_vis_not_avgd:
                                            if max(l) > maximumy: maximumy = max(l)
                                            if min(l) < minimumy: minimumy = min(l)
                                        plt.ylim([minimumy - 5, maximumy + 8])
                                    elif det == "PLT":
                                        maximumy = 370
                                        minimumy = 300  # 250
                                        for l in sigma_vis_not_avgd:
                                            if max(l) > maximumy: maximumy = max(l)
                                            if min(l) < minimumy: minimumy = min(l)
                                        plt.ylim([minimumy - 50, maximumy])
                                    elif det == "HFOC":
                                        plt.ylim([600, 1150])
                                    elif det == "HFET":
                                        plt.ylim([1600, 3600])
                                energy_str = "13.6 TeV" if energy == "13600" else "900 GeV"
                                plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
                                plt.text(0.28, 1.04, 'Work in progress', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
                                plt.text(0.87, 1.04, f'{energy_str}', ha='center', va='center', transform=ax.transAxes,weight="bold", size=14)
                                plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                                #if is_single_scan:
                                #    plt.text(0.03, 0.77, f'Single scan from Fill {fills[0]}', ha='left', va='center', transform=ax.transAxes,weight='bold', size=13)
                                #    plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_color_coded_bcid_{fills[0]}_{det}_{fit}_{corr}_{energy}.pdf")
                                #else:
                                plt.text(0.03, 0.77, f'Fill {fills[0]}, Scan {i+1} ', ha='left', va='center', transform=ax.transAxes, weight='bold', size=13)
                                plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_color_coded_bcid_{fills[0]}_scan{i+1}_{det}_{fit}_{corr}_{energy}.pdf")
                                plt.clf()
                                del f

                        if wantSigVisVsBCID:
                            for i in range(len(sbils_not_avgd)):  # looping over each scan
                                f = plt.figure(figsize=(8, 5))
                                ax = f.add_subplot(111)  # 1 by 1 grid of subplots, first subplot

                                # plotting the points in each scan
                                plt.scatter(bcids[i], sigma_vis_not_avgd[i], marker=".", alpha=1,s=10)
                                # plt.errorbar(bcids[i],sigma_vis_not_avgd[i], yerr=sigma_vis_err_not_avgd[i], alpha=1, ls='none', color = colors[i]) #errors are too small to make sense to plot

                                plt.xlabel('BCID', size=13)
                                plt.ylabel(r'$\sigma_{vis}$', size=13)
                                # adjusting the xlim
                                maximumx = 0
                                minimumx = 3
                                for l in bcids:
                                    if max(l) > maximumx: maximumx = max(l)
                                    if min(l) < minimumx: minimumx = min(l)
                                plt.xlim([minimumx - 0.5, maximumx + 0.5])
                                if energy == "13600":
                                    if det == "BCM1F":
                                        minimumy, maximumy = 140, 140
                                        for l in sigma_vis_not_avgd:
                                            if max(l) > maximumy: maximumy = max(l)
                                            if min(l) < minimumy: minimumy = min(l)
                                        plt.ylim([minimumy - 5, maximumy + 8])
                                    elif det == "PLT":
                                        maximumy = 370
                                        minimumy = 300  # 250
                                        for l in sigma_vis_not_avgd:
                                            if max(l) > maximumy: maximumy = max(l)
                                            if min(l) < minimumy: minimumy = min(l)
                                        plt.ylim([minimumy - 50, maximumy])
                                    elif det == "HFOC":
                                        plt.ylim([600, 1150])
                                    elif det == "HFET":
                                        plt.ylim([1600, 3600])
                                energy_str = "13.6 TeV" if energy == "13600" else "900 GeV"
                                plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes,weight="bold", size=14)
                                plt.text(0.28, 1.04, 'Work in progress', ha='center', va='center', transform=ax.transAxes, style='italic', size=13)
                                plt.text(0.87, 1.04, f'{energy_str}', ha='center', va='center', transform=ax.transAxes,weight="bold", size=14)
                                plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center',transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center',transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.77, f'Fill {fills[0]}, Scan {i + 1} ', ha='left', va='center',transform=ax.transAxes, weight='bold', size=13)
                                plt.savefig(f"{output_dir}/sigmavis_vs_BCID_{fills[0]}_scan{i + 1}_{det}_{fit}_{corr}_{energy}.pdf")
                                plt.clf()
                                del f



