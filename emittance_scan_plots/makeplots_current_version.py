import pandas as pd
import numpy as np
import shutil
import os
import argparse
import matplotlib.pyplot as plt
from statistics import mean

# make linearity plot for one fill, plotting sigvis vs sbil cc by scan (sigmavis_vs_SBIL_color_coded_scans). If input is one single fill.
# make plot of sigma vis as a function of train (sigmavis_vs_train), and also train length (sigmavis_vs_trainlength)
# see makeplots trains for more train stuff

#file currently: for each fill, sees if there are individual scans within, and averages over all the BCIDs within the one scan for an avgval.
#but beware, still implementing the effective fill number for the multiple scans in one fill to plot not all points on top of each other.

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('output', nargs='?', help='Output directory to store plots.')
    #args = parser.parse_args()

    output_dir = '/Users/leila/PycharmProjects/DESY/VDM' #args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dets = ["BCM1F"] # "HFOC", "HFET", "PLT", "BCM1F"]
    fits = ["DG"] # "SGConst", "DG", "SG","twoMuDG"]
    corrs = ["Background"] # "Background", "noCorr"
    energies = ["13600"]

    #fills_13TeV = [8007, 8016, 8017, 8027, 8030, 8033,8057,8058,8063,8067,8068]
    #fills_13TeV = [7966,7978,8007, 8016, 8027,8058,8063,8067,8068,8072,8078,8079,8081,8087,8088,8091,8094,8102,8103,8106,8112,8113,8115,8118,8120,8124,8128]
    fills_13TeV = [8088]
    blacklist_fill = []

    is_single_fill = True # if true, will make sigvis_vs_sbil_color_coded_scans
    wantSigVisVsTrain = False
    wantHistograms = False
    wantStandardPlots = False #sigvis vs fill and sigvis vs sbil. usually don't want.

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
                    traincounter = []
                    train_length =[] #list of the length of each train.
                    temp = []
                    new_fill_pos = [] #for sig vs train, need when new fill starts
                    for fill in fills:

                        if "HF" in det and fill == 7733:
                            continue
                        df_samefill = partial_df[partial_df["fill"] == fill]
                        #print(df_samefill)
                        if np.isnan(np.mean(df_samefill["xsec"])) or np.isnan(np.sqrt(df_samefill["SBILErr"].pow(2).sum())):
                            print(fill, ":(")
                            badfills.append(fill)
                            continue

                        new_fill_pos.append([len(temp), fill])  # for sig vs train, need when new fill starts, and name of fill
                        allscans_in_fill = list(np.asarray(df_samefill['scan_name'])) #gets all the scan names of the files of the same fill number
                        scans_in_fill = list(dict.fromkeys(allscans_in_fill)) #makes a list of the unique scan names from that. So if there were 2 scans in the fill, we would have 2 scan names

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

                            #for plot of sigvis vs train. want to see if BCID has IncOne (increased by one) in the successive row.
                            df_samescan['IncOne'] = (df_samescan["scan_name"] == df_samescan["scan_name"].shift())  # makes a new column in dataframe where first val is false, the rest true
                            df_samescan['IncOne'] = (np.where(df_samescan["IncOne"], np.where(df_samescan["BCID"].eq(df_samescan["BCID"].shift() + 1), 'true',
                                                                       df_samescan["BCID"] - df_samescan["BCID"].shift()),
                                                                        ''))  # where takes first a condition, then what the value should be replaced by if true, then val if false

                            new_train_indices = list(df_samescan.index[df_samescan["IncOne"]!='true'].values)  # gets the indices of the dataframe where a new train starts
                            avg_sigmavis_of_trains = []  # list of average sigvis in each train.

                            for i in range(len(new_train_indices)):
                                if new_train_indices[i] == new_train_indices[-1]:
                                    current_train_start_end = ([new_train_indices[i], df_samescan.index[-1] + 1])  # if we are at the index of the start of the last train in scan, set the start of the "next new train" to one after the end of the scan
                                else:
                                    current_train_start_end = new_train_indices[i:i + 2]  # gets a list containing 2 indices, the first one of the new train, and the index of the start of the next train
                                train_length.append(current_train_start_end[1]-current_train_start_end[0])
                                avg_sigmavis_of_trains.append(np.mean(df_samescan.loc[current_train_start_end[0]:current_train_start_end[1] - 1]['xsec']))  # loc slices based on index name, to get all the rows in current train. then avg the xsec

                            traincounter.append([new_train_indices, avg_sigmavis_of_trains]) #index or length of traincounter will give scan number.
                            temp.extend(avg_sigmavis_of_trains)


                            #making histograms for each scan
                            if wantHistograms:
                                f = plt.figure()
                                ax = f.add_subplot(111)
                                plt.hist(df_samescan["xsec"])
                                plt.xlabel(r'$\sigma_{vis}$')
                                plt.ylabel('count')
                                plt.text(0.28, 1.04, f'Scan: {scan}', ha='center', va='center', transform=ax.transAxes, style='italic', size=13)
                                plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.savefig(f"{output_dir}/sigmavis_histogram_per_scan_{scan}_{det}_{fit}_{corr}_{energy}.pdf")

                            #print(df_samescan, '\n', df_samescan.shape[0], df_samescan.shape[1]) #checking what shape returns for a df

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
                    if len(sigma_vis) > 0:
                        if wantStandardPlots:
                            ## Plot average SBIL vs. sigma_vis
                            f = plt.figure()
                            ax = f.add_subplot(111) #1 by 1 grid of subplots, first subplot.
                            plt.scatter(sbil, sigma_vis, marker=".", alpha = 1)
                            plt.errorbar(sbil, sigma_vis, yerr=sigma_vis_err, alpha = 0.1, ls='none') #fmt="o"
                            plt.xlabel(r'SBIL [$Hz/\mu b$]', size=13)
                            plt.ylabel(r'$\sigma_{vis} \,[\mu b]$', size=13)
                            plt.xlim([0, 5])
                            if any(val > 5 for val in sbil): plt.xlim([0,max(sbil)+1])
                            if energy == "13600":
                                if det == "BCM1F":
                                      plt.ylim([100, 180])
                                elif det == "PLT":
                                      plt.ylim([250, 370])
                                elif det == "HFOC":
                                      plt.ylim([600, 1150])
                                elif det == "HFET":
                                      plt.ylim([1600, 3600])
                            energy_str = "13.6 TeV" if energy=="13600" else "900 GeV"
                            plt.text(0.06, 1.04,'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.28, 1.04,'Work in progress', ha='center', va='center', transform=ax.transAxes, style='italic', size=13)
                            plt.text(0.87, 1.04,r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes, size=14)
                            plt.text(0.03, 0.95,f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.89,f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.83,f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes, size=13)
                            if is_single_fill:
                                plt.text(0.03, 0.77, f'Fill {fills[0]}', ha='left', va='center', transform=ax.transAxes, size=13)
                                plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_{fills[0]}_{det}_{fit}_{corr}_{energy}.pdf")
                            else: plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_{det}_{fit}_{corr}_{energy}.pdf")
                            plt.clf()
                            del f

                            ## Plot scan average vs. sigma_vis
                            f = plt.figure()
                            ax = f.add_subplot(111)
                            plt.scatter(eff_fills, sigma_vis, marker=".", alpha = 1)
                            plt.errorbar(eff_fills, sigma_vis, yerr=sigma_vis_err, fmt="o", alpha= 1)
                            plt.xlabel('Fill', size=13)
                            plt.ylabel(r'$\sigma_{vis}$', size=13)
                            plt.text(0.06, 1.04,'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.28, 1.04,'Work in progress', ha='center', va='center', transform=ax.transAxes, style='italic', size=13)
                            plt.text(0.87, 1.04,r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes, size=14)
                            plt.text(0.03, 0.95,f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.89,f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.83,f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.xlim([np.min(eff_fills)-5, np.max(eff_fills)+5])
                            if energy == "13600":
                                if det == "BCM1F":
                                      plt.ylim([100, 180])
                                elif det == "PLT":
                                      plt.ylim([250, 370])
                                elif det == "HFOC":
                                      plt.ylim([600, 1150])
                                elif det == "HFET":
                                      plt.ylim([1600, 3600])
                            if is_single_fill:
                                plt.text(0.03, 0.77, f'Fill {fills[0]}', ha='left', va='center', transform=ax.transAxes, size=13)
                                plt.savefig(f"{output_dir}/sigmavis_vs_fill_{fills[0]}_{det}_{fit}_{corr}_{energy}.pdf")
                            else: plt.savefig(f"{output_dir}/sigmavis_vs_fill_{det}_{fit}_{corr}_{energy}.pdf")
                            plt.clf()
                            del f

                        # now if we had multiple scans in the fill, we want to plot sigma vis as fkt of SBIL again but now with all the SBIL values just color coded for each scan
                        if is_single_fill: #and len(sbils_not_avgd)>1:
                            f = plt.figure()
                            ax = f.add_subplot(111)  # 1 by 1 grid of subplots, first subplot.
                            colors = ['red','orange','gold','yellowgreen','green','deepskyblue','blue','indigo'] #maximum number of scans at the moment is 7
                            for i in range(len(sbils_not_avgd)): #looping over each scan
                                # making linear fit for each scan
                                #linfit, cov = np.polyfit(sbils_not_avgd[i], sigma_vis_not_avgd[i], 1, cov=True)
                                #plt.plot(sbils_not_avgd[i], [element * linfit[0] + linfit[1] for element in sbils_not_avgd[i]],color=colors[i])

                                ''''''
                                #To get rid of the huge outliers. Necessary for twoMuDG 8007,8030,8033,8057,8078,8079,8081,8088,8103,8106. (see makeplots bcid to investigate the points)
                                point = 0 
                                huge_points =[]
                                while point < len(sigma_vis_not_avgd[i])-1 :#looping over the index of all the elements in the list of sigma vis for one scan.
                                    if sigma_vis_not_avgd[i][point] > 500 or sbils_not_avgd[i][point]>20: #6000
                                        huge_points.append(sigma_vis_not_avgd[i].pop(point))
                                        sbils_not_avgd[i].pop(point)
                                    else: point+=1
                                print("HUGE POINTS REMOVED:", huge_points)
                                ''''''

                                # linear fit with avg SBIL in order to decorrelate the 2 fit parameters
                                avg_sbil_of_scan = mean(sbils_not_avgd[i])
                                linfit, cov = np.polyfit(sbils_not_avgd[i]-avg_sbil_of_scan, sigma_vis_not_avgd[i], 1, cov = True)
                                plt.plot(sbils_not_avgd[i], [element * linfit[0] - avg_sbil_of_scan* linfit[0] + linfit[1] for element in sbils_not_avgd[i]], color=colors[i]) #- avg_sbil_of_scan* linfit[0] is to make line correct w pts
                                err = np.sqrt(np.diag(cov))

                                #plotting the points in each scan
                                if i == 0:
                                    plt.scatter(sbils_not_avgd[0], sigma_vis_not_avgd[0], marker=".", alpha=1, color='red', label='first scan', s=10)
                                #plt.scatter(sbils_not_avgd[i],sigma_vis_not_avgd[i], marker=".", alpha=1, color = colors[i], s=10, label = f'({round(linfit[0],2)}$\pm${round(err[0],2)})x + {round(linfit[1],2)}$\pm${round(err[1],2)}') #for normal fit
                                plt.scatter(sbils_not_avgd[i],sigma_vis_not_avgd[i], marker=".", alpha=1, color = colors[i], s=10, label = f'({round(linfit[0],2)}$\pm${round(err[0],2)})(x-avgSBIL) + {round(linfit[1],2)}$\pm${round(err[1],2)}') #for fit with decorrelated params
                                #plt.errorbar(sbils_not_avgd[i],sigma_vis_not_avgd[i], yerr=sigma_vis_err_not_avgd[i], alpha=1, ls='none', color = colors[i])

                            plt.xlabel(r'SBIL [$Hz/\mu b$]', size=15)
                            plt.ylabel(r'$\sigma_{vis} \,[\mu b]$', size=15)
                            plt.legend(fontsize = 7.5) #loc = 3 is bottom left
                            #adjusting the xlim
                            maximumx = 0
                            minimumx = 3
                            for l in sbils_not_avgd:
                                if max(l)>maximumx: maximumx=max(l)
                                if min(l)<minimumx: minimumx=min(l)
                            plt.xlim([minimumx-0.5, maximumx+0.5])
                            if energy == "13600":
                                if det == "BCM1F":
                                    minimumy, maximumy = 140,140
                                    for l in sigma_vis_not_avgd:
                                        if max(l) > maximumy: maximumy = max(l)
                                        if min(l) < minimumy: minimumy = min(l)
                                    plt.ylim([minimumy-5, maximumy+10])
                                elif det == "PLT":
                                    maximumy = 400
                                    minimumy = 300
                                    for l in sigma_vis_not_avgd:
                                        if max(l) > maximumy: maximumy = max(l)
                                        if min(l) < minimumy: minimumy = min(l)
                                    plt.ylim([minimumy-5, maximumy+10])
                                elif det == "HFOC":
                                    plt.ylim([600, 1150])
                                elif det == "HFET":
                                    plt.ylim([1600, 3600])
                            energy_str = "13.6 TeV" if energy == "13600" else "900 GeV"

                            plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
                            plt.text(0.28, 1.04, 'Work in progress', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
                            plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes, size=14)
                            plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.77, f'Color coded by scan', ha='left', va='center', transform=ax.transAxes,weight='bold', size=13)
                            plt.text(0.03, 0.71, f'Fill {fills[0]}', ha='left', va='center', transform=ax.transAxes,size=13)

                            '''
                            #official format for joanna
                            plt.text(0.03, 0.95, 'CMS', ha='left', va='center', transform=ax.transAxes, weight="bold", size=14)
                            plt.text(0.14, 0.95, 'Preliminary', ha='left', va='center', transform=ax.transAxes, style='italic', size=13)
                            plt.text(1, 1.04, f'Fill {fills[0]} (2022, {energy_str})', ha='right', va='center', transform=ax.transAxes, size=14)
                            plt.text(0.03, 0.89, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.83, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.77, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.71, f'Color coded by scan', ha='left', va='center', transform=ax.transAxes, weight='bold', size=13)
                            '''

                            plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_color_coded_scans_{fills[0]}_{det}_{fit}_{corr}_{energy}.pdf")
                            #plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_color_coded_scans_{fills[0]}_{det}_{fit}_{corr}_{energy}_constrained_fit.pdf")
                            #plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_color_coded_scans_{fills[0]}_{det}_{fit}_{corr}_{energy}_reprocessed.pdf")
                            plt.clf()
                            del f

                        if wantSigVisVsTrain:
                            # plot of sigma vis as a function of individual train
                            #traincounter is a list of lists. each inner list is one scan, containing 2 lists (first being index of trainstarts, second being avg sigvis over that train)
                            num_scans = len(traincounter)
                            new_scan_pos =[]
                            sigma_vis_train_avg =[] #single list of all the average sigvis in each train regardless of scan
                            for scan in traincounter:
                                new_scan_pos.append(len(sigma_vis_train_avg))
                                sigma_vis_train_avg.extend(scan[1])

                            num_trains = len(sigma_vis_train_avg)
                            enumerate_trains = list(range(num_trains))

                            f = plt.figure(figsize=(10,6))
                            ax = f.add_subplot(111)
                            plt.vlines(x=[x for x in new_scan_pos], ymin=0, ymax=4000, colors=['purple'],ls='--', lw=1, alpha=0.5, label = "new scan")
                            plt.vlines(x=[x[0] for x in new_fill_pos], ymin=0, ymax=4000, colors=['red'],ls='-', lw=1, alpha=0.5, label = f'new fill')

                            plt.scatter(enumerate_trains, sigma_vis_train_avg, c=train_length, cmap='viridis', vmin=0, vmax=max(train_length), marker=".", alpha=1, s=100, zorder=4)
                            cbar = plt.colorbar()
                            cbar.set_label('Train Length', rotation=270, labelpad=+15)
                            plt.legend()
                            plt.xlabel('Train', size=13)
                            plt.ylabel(r'$\sigma_{vis}$', size=13)
                            plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
                            plt.text(0.28, 1.04, 'Work in progress', ha='center', va='center', transform=ax.transAxes, style='italic', size=13)
                            plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes, size=14)
                            plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes, size=13)
                            plt.text(0.03, 0.77, f'Average over BCIDs for each train', ha='left', va='center',weight='bold',transform=ax.transAxes, size=13)
                            # plt.text(0.03, 0.71, f'band shows avg and rmse', ha='left', va='center',transform=ax.transAxes, size=13)

                            plt.xlim([np.min(enumerate_trains) - 1, np.max(enumerate_trains) + 1])
                            if energy == "13600":
                                if det == "BCM1F":
                                    plt.ylim([100, 180])
                                elif det == "PLT":
                                    plt.ylim([250, 370])
                                elif det == "HFOC":
                                    plt.ylim([600, 1200])
                                elif det == "HFET":
                                    plt.ylim([2400, 4000])
                            height = 120
                            if det == "HFOC": height = 700
                            if det == "HFET": height = 2700
                            for x in new_fill_pos:
                                xpos = x[0]
                                plt.text(xpos, height, f'FILL {x[1]}', horizontalalignment='left', verticalalignment='center', rotation='vertical')
                            plt.savefig(f"{output_dir}/sigmavis_vs_train_{det}_{fit}_{corr}_{energy}.pdf")
                            plt.clf()
                            del f

                            #plot of sigvis vs train length
                            print()
                            f = plt.figure(figsize=(8,5))
                            ax = f.add_subplot(111)
                            plt.scatter(train_length, sigma_vis_train_avg, marker=".", alpha=1, s=100, zorder=4, label=f'trains from fills \n{", ".join(repr(e) for e in fills[0:5])}\n{", ".join(repr(e) for e in fills[5:])}')
                            plt.legend(ncol=2)
                            plt.xlabel('Train Length', size=13)
                            plt.ylabel(r'$\sigma_{vis}$', size=13)
                            plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
                            plt.text(0.28, 1.04, 'Work in progress', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
                            plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes,weight="bold", size=14)
                            plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                            plt.text(0.03, 0.77, f'Average over BCIDs for each train', ha='left', va='center',weight='bold',transform=ax.transAxes, size=13)
                            # plt.text(0.03, 0.71, f'band shows avg and rmse', ha='left', va='center',transform=ax.transAxes, size=13)

                            plt.xlim([np.min(train_length) - 1, np.max(train_length) + 1])
                            if energy == "13600":
                                if det == "BCM1F":
                                    plt.ylim([100, 180])
                                elif det == "PLT":
                                    plt.ylim([250, 370])
                                elif det == "HFOC":
                                    plt.ylim([600, 1200])
                                elif det == "HFET":
                                    plt.ylim([2400, 4000])
                            height = 120
                            if det == "HFOC": height = 700
                            if det == "HFET": height = 2700
                            plt.savefig(f"{output_dir}/sigmavis_vs_trainlength_{det}_{fit}_{corr}_{energy}.pdf")
                            plt.clf()
                            del f

