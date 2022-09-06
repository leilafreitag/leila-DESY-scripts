import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from statistics import mean

# makes plot of sigvis vs SBIL and color codes leading bunches, etc. (sigmavis_vs_SBIL_color_coded_trains)
# makes plot of sigvis vs BCID with each train in a different color (sigmavis_vs_BCID_color_coded_trains)
# also prints the BCIDs of the leading bunches and the last bunches in each train

if __name__ == "__main__":

    output_dir = '/Users/leila/PycharmProjects/DESY/VDM' #CHANGE THIS
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dets = ["BCM1F"] # "HFOC", "HFET", "PLT", "BCM1F"]
    fits = ["SG"] # "SGConst", "DG", "SG"]
    corrs = ["Background"] # "Background", "noCorr"
    energies = ["13600"]

    #fills_13TeV = [8007, 8016, 8027, 8058, 8063, 8067, 8068, 8072, 8078, 8079, 8081, 8087, 8088, 8091, 8094, 8102, 8103, 8106, 8112, 8113, 8115, 8118]
    fills_13TeV = [8088]
    blacklist_fill = []

    wantSigVisVsSBIL_ColorCodeLeadingBunches = True
    wantSigVisVsBCID_ColorCodeTrains = True
    is_single_fill = True

    file = "fit_and_lumidata.csv"
    df = pd.read_csv(file, sep=',')
    if file == "fit_and_lumidata_reprocessed.csv": reprocessed_file = True
    else: reprocessed_file = False

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
                    traincounter_leading = [] #list of lists for making plot with leading bunches
                    traincounter_nonleading =[]
                    traincounter_nextleading =[]
                    traincounter_end =[]
                    train_length =[] #list of the length of each train.
                    new_fill_pos = [] #for sig vs train, need when new fill starts

                    for fill in fills:

                        if "HF" in det and fill == 7733:
                            continue
                        df_samefill = partial_df[partial_df["fill"] == fill]
                        if np.isnan(np.mean(df_samefill["xsec"])) or np.isnan(np.sqrt(df_samefill["SBILErr"].pow(2).sum())):
                            print(fill, ":(")
                            badfills.append(fill)
                            continue

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
                            bcids_current_scan = list(np.asarray(df_samescan["BCID"]))
                            bcids.append(bcids_current_scan)

                            #for plot of sigvis vs train. want to see if BCID has IncOne (increased by one) in the successive row.
                            df_samescan['IncOne'] = (df_samescan["scan_name"] == df_samescan["scan_name"].shift())  # makes a new column in dataframe where first val is false, the rest true
                            df_samescan['IncOne'] = (np.where(df_samescan["IncOne"], np.where(df_samescan["BCID"].eq(df_samescan["BCID"].shift() + 1), 'true',
                                                                       df_samescan["BCID"] - df_samescan["BCID"].shift()), ''))  # where takes first a condition, then what the value should be replaced by if true, then val if false
                            new_train_indices = list(df_samescan.index[df_samescan["IncOne"]!='true'].values)  # gets the indices of the dataframe where a new train starts
                            print(new_train_indices)
                            sigmavis_of_trains = []  # list of sigvis in each train
                            sbils_of_leading_bunches =[] #list of sbils for each train
                            sbils_of_nextleading_bunches = []
                            sbils_of_nonleading_bunches = []
                            sbils_of_end_bunches = []
                            sigmavis_of_leading_bunches = []
                            sigmavis_of_nextleading_bunches =[]
                            sigmavis_of_nonleading_bunches = []
                            sigmavis_of_end_bunches =[]
                            bcids_of_leading_bunches =[]
                            bcids_of_nonleading_bunches =[]
                            bcids_of_last_bunches = []
                            train_enumerator_leading_bunches = []  # list of 1s for every bunch in trains, but leading bunches are 0
                            train_enumerator = []  # list of numbers counting the train number, repeated as many times as there are bcids in train. ie 11112222222233333

                            n=1
                            for i in range(len(new_train_indices)):
                                if new_train_indices[i] == new_train_indices[-1]:
                                    current_train_start_end = ([new_train_indices[i], df_samescan.index[-1] + 1])  # if we are at the index of the start of the last train in scan, set the start of the "next new train" to one after the end of the scan
                                else:
                                    current_train_start_end = new_train_indices[i:i + 2]  # gets a list containing 2 indices, the first one of the new train, and the index of the start of the next train
                                train_length.append(current_train_start_end[1]-current_train_start_end[0])

                                sbils_of_trains_current = list(df_samescan.loc[current_train_start_end[0]:current_train_start_end[1] - 1]['SBIL'])
                                sbils_of_leading_bunches.append(sbils_of_trains_current[0])
                                sbils_of_nonleading_bunches.extend(sbils_of_trains_current[1:])

                                sigmavis_of_trains_current = list(df_samescan.loc[current_train_start_end[0]:current_train_start_end[1] - 1]['xsec'])
                                sigmavis_of_leading_bunches.append(sigmavis_of_trains_current[0])
                                sigmavis_of_nonleading_bunches.extend(sigmavis_of_trains_current[1:])
                                sigmavis_of_trains.append(sigmavis_of_trains_current)

                                bcids_of_trains_current = list(df_samescan.loc[current_train_start_end[0]:current_train_start_end[1] - 1]['BCID'])
                                bcids_of_leading_bunches.append(bcids_of_trains_current[0])
                                bcids_of_nonleading_bunches.extend(bcids_of_trains_current[1:])
                                bcids_of_last_bunches.append(bcids_of_trains_current[-1])


                                if (current_train_start_end[1]-current_train_start_end[0])>5: #if train long enough to split also into nextleading bunches
                                    sbils_of_nextleading_bunches.append(sbils_of_trains_current[1:4]) #END DEFINES HOW MANY NEXTLEADING BUNCHES I CAN PLOT IN COLOR
                                    sigmavis_of_nextleading_bunches.append(sigmavis_of_trains_current[1:4])
                                    sbils_of_end_bunches.append(sbils_of_trains_current[-2:])
                                    sigmavis_of_end_bunches.append(sigmavis_of_trains_current[-2:])

                                temporary = list(np.full_like(sigmavis_of_trains_current, 1, dtype=np.double)) #list of 1 for every bcid in train
                                temporary[0] = 0 #set first bcid to 0 because its leading bunch
                                train_enumerator_leading_bunches.extend(temporary)
                                train_enumerator.extend(list(np.full_like(sigmavis_of_trains_current, n, dtype=np.double))) #EXTEND instead of append here because i want one list of trainnumberextended
                                n+=1

                            traincounter.append([new_train_indices, sigmavis_of_trains,train_enumerator,train_enumerator_leading_bunches]) #index or length of traincounter will give scan number.
                            traincounter_leading.append([sbils_of_leading_bunches,sigmavis_of_leading_bunches,bcids_of_leading_bunches])
                            traincounter_nonleading.append([sbils_of_nonleading_bunches,sigmavis_of_nonleading_bunches,bcids_of_nonleading_bunches])
                            traincounter_nextleading.append([sbils_of_nextleading_bunches,sigmavis_of_nextleading_bunches])
                            traincounter_end.append([sbils_of_end_bunches,sigmavis_of_end_bunches,bcids_of_last_bunches])

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

                    #color coded plots
                    if is_single_fill: #and len(sbils_not_avgd)>1:
                        '''
                        for i in range(len(sbils_not_avgd)):  # looping over each scan to remove extreme outliers. FOR THE twoMuDG fit.
                            # To get rid of the huge outliers.(see makeplots current version). Here want to get BCID of these outliers to investigate
                            # Necessary for twoMuDG 8007,8030,8033,8057,8078,8079,8081,8088,8103,8106.
                            point = 0
                            huge_points = []
                            while point < len(sigma_vis_not_avgd[i]) - 1:  # looping over the index of all the elements in the list of sigma vis for one scan.
                                if sigma_vis_not_avgd[i][point] > 500 or sbils_not_avgd[i][point] > 20:  # 6000
                                    huge_points.append(bcids[i].pop(point))
                                    huge_points.append(sigma_vis_not_avgd[i].pop(point))
                                    sbils_not_avgd[i].pop(point)
                                    traincounter[i][2].pop(point)
                                else:
                                    point += 1
                            print("HUGE POINTS REMOVED (bcid followed by its sigmavis):", huge_points)
                        '''
                        if wantSigVisVsSBIL_ColorCodeLeadingBunches:
                            for i in range(len(sbils_not_avgd)):  # looping over each scan
                                f = plt.figure(figsize=(6.6, 4.8))
                                ax = f.add_subplot(111)  # 1 by 1 grid of subplots, first subplot
                                # making linear fit for each scan
                                # linfit, cov = np.polyfit(sbils_not_avgd[i], sigma_vis_not_avgd[i], 1, cov=True)
                                # plt.plot(sbils_not_avgd[i], [element * linfit[0] + linfit[1] for element in sbils_not_avgd[i]],color=colors[i])

                                # linear fit with avg SBIL in order to decorrelate the 2 fit parameters
                                avg_sbil_of_scan = mean(sbils_not_avgd[i])
                                linfit, cov = np.polyfit(sbils_not_avgd[i] - avg_sbil_of_scan, sigma_vis_not_avgd[i], 1,cov=True)
                                err = np.sqrt(np.diag(cov))
                                plt.plot(sbils_not_avgd[i],[element * linfit[0] - avg_sbil_of_scan * linfit[0] + linfit[1] for element in sbils_not_avgd[i]], color='black',
                                         label=f'({round(linfit[0], 2)}$\pm${round(err[0], 2)})(x-avgSBIL) + {round(linfit[1], 2)}$\pm${round(err[1], 2)}',zorder=0)  # - avg_sbil_of_scan* linfit[0] is to make line correct w pts

                                # plotting the points in each scan (SIGVIS vs SBIL, color dictated by leading or nonleading bunch)
                                plt.scatter(traincounter_leading[i][0], traincounter_leading[i][1], color = "red", label = "leading bunches", marker=".", alpha=1,s=13,zorder=5)
                                for s in range(len(traincounter_nextleading[i][0])): #s loops over each train.
                                    if s==0:
                                        plt.scatter(traincounter_nextleading[i][0][s][0], traincounter_nextleading[i][1][s][0],color = 'gold',label = "second bunches", marker=".", alpha=1,s=13,zorder=4)
                                        plt.scatter(traincounter_nextleading[i][0][s][1], traincounter_nextleading[i][1][s][1],color = 'yellowgreen',label = "third bunches", marker=".", alpha=1,s=13,zorder=3)
                                        plt.scatter(traincounter_nextleading[i][0][s][2], traincounter_nextleading[i][1][s][2],color = 'deepskyblue',label = "fourth bunches", marker=".", alpha=1,s=13,zorder=2)
                                    plt.scatter(traincounter_nextleading[i][0][s][0], traincounter_nextleading[i][1][s][0],color = 'gold', marker=".", alpha=1,s=13,zorder=4)
                                    plt.scatter(traincounter_nextleading[i][0][s][1], traincounter_nextleading[i][1][s][1],color = 'yellowgreen', marker=".", alpha=1,s=13,zorder=3)
                                    plt.scatter(traincounter_nextleading[i][0][s][2], traincounter_nextleading[i][1][s][2],color = 'deepskyblue', marker=".", alpha=1,s=13, zorder=2)
                                plt.scatter(traincounter_nonleading[i][0], traincounter_nonleading[i][1], color = "blue", label = "train bunches", marker=".", alpha=1,s=13, zorder=1)

                                plt.scatter(traincounter_end[i][0], traincounter_end[i][1],color = 'black',label="last and second to last bunches", marker=".", alpha=1,s=13,zorder=4)

                                # plt.errorbar(sbils_not_avgd[i],sigma_vis_not_avgd[i], yerr=sigma_vis_err_not_avgd[i], alpha=1, ls='none', color = colors[i])

                                #if want to COLORCODE BY TRAIN:
                                #plt.scatter(sbils_not_avgd[i], sigma_vis_not_avgd[i], c=traincounter[i][2], cmap='viridis', vmin=min(traincounter[i][2]),vmax=max(traincounter[i][2]), marker=".", alpha=1,s=10) #if want to colorcode by train
                                #cbar = plt.colorbar()
                                #cbar.set_label('Train', rotation=270, labelpad=+15)

                                print(f"bcids of leading bunches in scan {i+1}:", traincounter_leading[i][2])
                                print(f"bcids of last bunches in scan {i + 1}:", traincounter_end[i][2])
                                print(f'number of leading bunches:', len(traincounter_leading[i][0]))
                                print(f'number of nextleading bunches:', len(traincounter_nextleading[i][0]))
                                print(f'number of end bunches:', len(traincounter_end[i][0]) * len(traincounter_end[i][0][0])) #number of trains times number of "end" bunches saved in each train

                                plt.xlabel(r'SBIL [$Hz/\mu b$]', size=13)
                                plt.ylabel(r'$\sigma_{vis} \,[\mu b]$', size=13)
                                plt.legend(fontsize = 7.5,loc=1) #loc = 3 is bottom left
                                #adjusting the xlim
                                minimumx, maximumx = 3, 0
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
                                        minimumy, maximumy = 300,400
                                        for l in sigma_vis_not_avgd:
                                            if max(l) > maximumy: maximumy = max(l)
                                            if min(l) < minimumy: minimumy = min(l)
                                        plt.ylim([minimumy-5, maximumy+10])
                                    elif det == "HFOC":
                                        minimumy, maximumy = 1150, 600
                                        for l in sigma_vis_not_avgd:
                                            if max(l) > maximumy: maximumy = max(l)
                                            if min(l) < minimumy: minimumy = min(l)
                                        plt.ylim([minimumy - 5, maximumy + 10])
                                    elif det == "HFET":
                                        minimumy, maximumy = 3600, 1600
                                        for l in sigma_vis_not_avgd:
                                            if max(l) > maximumy: maximumy = max(l)
                                            if min(l) < minimumy: minimumy = min(l)
                                        plt.ylim([minimumy - 5, maximumy + 10])
                                energy_str = "13.6 TeV" if energy == "13600" else "900 GeV"
                                plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
                                plt.text(0.23, 1.04, 'Preliminary', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
                                plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes, size=14)
                                plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                                plt.text(0.03, 0.77, f'Fill {fills[0]}, Scan {i + 1} ', ha='left', va='center',transform=ax.transAxes, weight='bold', size=13)
                                if reprocessed_file:
                                    plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_color_coded_trains_{fills[0]}_scan{i + 1}_{det}_{fit}_{corr}_{energy}_reprocessed.pdf")
                                else:
                                    plt.savefig(f"{output_dir}/sigmavis_vs_SBIL_color_coded_trains_{fills[0]}_scan{i + 1}_{det}_{fit}_{corr}_{energy}.pdf")
                                plt.clf()
                                del f

                        if wantSigVisVsBCID_ColorCodeTrains:
                            for i in range(len(sbils_not_avgd)):  # looping over each scan
                                f = plt.figure(figsize=(6.6, 4.8))
                                ax = f.add_subplot(111)  # 1 by 1 grid of subplots, first subplot

                                # plt.errorbar(bcids[i],sigma_vis_not_avgd[i], yerr=sigma_vis_err_not_avgd[i], alpha=1, ls='none', color = colors[i]) #errors are too small to make sense to plot

                                plt.scatter(bcids[i], sigma_vis_not_avgd[i], c=traincounter[i][2], cmap='prism', vmin=min(traincounter[i][2]),vmax=max(traincounter[i][2]), marker=".", alpha=1,s=10) #if want to colorcode by train
                                cbar = plt.colorbar()
                                cbar.set_label('Train', rotation=270, labelpad=+15)

                                plt.xlabel('BCID', size=13)
                                plt.ylabel(r'$\sigma_{vis} \,[\mu b]$', size=13)
                                # adjusting the xlim
                                maximumx = 750
                                minimumx = 0
                                #for l in bcids:
                                #    if max(l) > maximumx: maximumx = max(l)
                                #    if min(l) < minimumx: minimumx = min(l)
                                plt.xlim([minimumx - 0.5, maximumx + 0.5])
                                if energy == "13600":
                                    if det == "BCM1F":
                                        minimumy, maximumy = 140, 140
                                        for l in sigma_vis_not_avgd:
                                            if max(l) > maximumy: maximumy = max(l)
                                            if min(l) < minimumy: minimumy = min(l)
                                        plt.ylim([minimumy - 3, maximumy + 9])
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
                                plt.text(0.26, 1.04, 'Preliminary', ha='center', va='center',transform=ax.transAxes, style='italic', size=13)
                                plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center',transform=ax.transAxes, size=14)
                                plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center',transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center',transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center',transform=ax.transAxes, size=13)
                                plt.text(0.03, 0.77, f'Fill {fills[0]}, Scan {i + 1} ', ha='left', va='center', transform=ax.transAxes, weight='bold', size=13)
                                plt.text(0.03, 0.71, f'Zoomed in to show only first few trains', ha='left', va='center',transform=ax.transAxes, size=13)

                                if reprocessed_file:
                                    plt.savefig(f"{output_dir}/sigmavis_vs_BCID_color_coded_trains_{fills[0]}_scan{i + 1}_{det}_{fit}_{corr}_{energy}_reprocessed.pdf")
                                else:
                                    plt.savefig(f"{output_dir}/sigmavis_vs_BCID_color_coded_trains_{fills[0]}_scan{i + 1}_{det}_{fit}_{corr}_{energy}.pdf")
                                plt.clf()
                                del f

