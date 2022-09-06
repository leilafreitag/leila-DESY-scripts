import pandas as pd
import numpy as np
import shutil
import os
import argparse
import matplotlib.pyplot as plt
from statistics import mean

# makes stability plot of sigvis vs scan only for leading bunches (sigmavis_vs_scan_leading_bunches)
# makes plot sigvis for each leading bunch in a train (sigmavis_vs_train_leading_bunches)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('output', nargs='?', help='Output directory to store plots.')
    #args = parser.parse_args()

    output_dir = '/Users/leila/PycharmProjects/DESY/VDM' #args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dets = ["BCM1F"] # "HFOC", "HFET", "PLT", "BCM1F"]
    fits = ["DG"] # "SGConst", "DG", "SG"]
    corrs = ["Background"] # "Background", "noCorr"
    energies = ["13600"]

    #fills_13TeV = [7679, 7725, 7816, 7886, 8007]
    #fills_13TeV = [8007, 8016, 8017, 8027, 8030, 8033,8057,8058,8063,8067,8068]
    #fills_13TeV = [8007, 8016, 8027,8058,8063,8067,8068,8072,8078,8079,8081,8087,8088,8094]
    #fills_13TeV = [8007, 8016, 8027,8058,8063,8067,8068,8072,8078,8079,8081,8087,8088,8091,8094,8102,8103,8106,8112,8113,8115,8118,8120,8124,8128]
    fills_13TeV = [7966,7978,8007, 8016, 8027,8058,8063,8067,8068,8072,8078,8079,8081,8087,8088,8091,8094,8102,8103,8106,8112,8113,8115,8118,8120,8124,8128,8132,8136,8142,8143,8144,8146,8147,8148,8149,8151] #7920, 7921,7960,7963,

    blacklist_fill = []

    wantSigVisVsScan_LeadingBunches = True
    is_single_fill = True

    df = pd.read_csv("fit_and_lumidata.csv", sep=',')

    sigma_vis = df["xsec"]


    def rmse(theoretical, data):
        return np.sqrt(mean((theoretical - data) ** 2))

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
                    fillcounter = [] #for the plot of sigma vis as a fkt of scan, to know where to draw the vertical lines separating diff fills

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
                    train_count = [] #will be list of lists. each list will be a scan. within each scan list count trains like [1,2,3,4],[5,6,7] for 2 scans.
                    train_count_index = 1
                    num_scans_in_fill =[]

                    for fill in fills:

                        if "HF" in det and fill == 7733:
                            continue
                        df_samefill = partial_df[partial_df["fill"] == fill]

                        if np.isnan(np.mean(df_samefill["xsec"])) or np.isnan(np.sqrt(df_samefill["SBILErr"].pow(2).sum())):
                            print(fill, ":(")
                            badfills.append(fill)
                            continue

                        fillcounter.append([len(sigma_vis),fill])  # the length of sigma vis should give me the (scan) index the fill starts at, and what fill it is

                        allscans_in_fill = list(np.asarray(df_samefill['scan_name'])) #gets all the scan names of the files of the same fill number
                        scans_in_fill = list(dict.fromkeys(allscans_in_fill)) #makes a list of the unique scan names from that. So if there were 2 scans in the fill, we would have 2 scan names
                        num_scans_in_fill.append(len(scans_in_fill))

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
                            train_enumerator_leading_bunches = []  # list of 1s for every bunch in trains, but leading bunches are 0
                            train_enumerator = []  # list of numbers counting the train number, repeated as many times as there are bcids in train. ie 11112222222233333
                            train_count_per_scan = [] #list for each scan, with overall train number. for example one scan could have trains [5,6,7]
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
                                train_count_per_scan.append(train_count_index)
                                train_count_index+=1

                            traincounter.append([new_train_indices, sigmavis_of_trains,train_enumerator,train_enumerator_leading_bunches]) #index or length of traincounter will give scan number.
                            traincounter_leading.append([sbils_of_leading_bunches,sigmavis_of_leading_bunches,bcids_of_leading_bunches])
                            traincounter_nonleading.append([sbils_of_nonleading_bunches,sigmavis_of_nonleading_bunches,bcids_of_nonleading_bunches])
                            traincounter_nextleading.append([sbils_of_nextleading_bunches,sigmavis_of_nextleading_bunches])
                            traincounter_end.append([sbils_of_end_bunches,sigmavis_of_end_bunches])
                            train_count.append(train_count_per_scan)

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
                    #print("BCIDs: ", bcids)

                    #color coded plots
                    if len(sigma_vis) > 0 and wantSigVisVsScan_LeadingBunches:
                        energy_str = "13.6 TeV" if energy == "13600" else "900 GeV"

                        # plot of sigma vis of leading bunches as a function of train.
                        num_scans = len(traincounter)  # traincounter is a list of lists. each inner list is one scan, containing 2 lists (first being index of trainstarts, second being avg sigvis over that train)
                        new_scan_pos = []
                        sigma_vis_train_avg = []  # single list of all the average sigvis in each train regardless of scan
                        for scan in traincounter:
                            new_scan_pos.append(len(sigma_vis_train_avg)+1)
                            sigma_vis_train_avg.extend(scan[1])

                        num_trains = len(sigma_vis_train_avg)
                        enumerate_trains = list(range(num_trains))
                        #print(mean(sigma_vis_train_avg))
                        f = plt.figure(figsize=(8,5))
                        ax = f.add_subplot(111)
                        plt.vlines(x=[x-0.5 for x in new_scan_pos], ymin=0, ymax=4000, colors=['purple'], ls='--', lw=1,alpha=0.5, label="new scan")

                        new_fill_pos_in_trains = [] #want the fill position in (train) index not in (scan) index.
                        for n in fillcounter: #n is a list for each fill, with scan index fill starts at, and fill name.
                            new_fill_pos_in_trains.append(new_scan_pos[n[0]])
                        plt.vlines(x=[x-0.5 for x in new_fill_pos_in_trains], ymin=0, ymax=4000, colors=['red'], ls='-', lw=1,alpha=0.5, label=f'new fill')

                        for i in range(num_scans):  # looping over each scan
                            plt.scatter(train_count[i], traincounter_leading[i][1], marker=".", color = 'blue', alpha=1, s=100, zorder=4)

                        plt.legend()
                        plt.xlabel('Train', size=13)
                        plt.ylabel(r'$\sigma_{vis}$', size=13)
                        plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
                        plt.text(0.20, 1.04, 'Preliminary', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
                        plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes,weight="bold", size=14)
                        plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes,size=13)
                        plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes,size=13)
                        plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,size=13)
                        plt.text(0.03, 0.77, f'Leading bunches only', ha='left', va='center', weight='bold', transform=ax.transAxes, size=13)

                        plt.xlim([1 - 1, train_count_index + 1])
                        if energy == "13600":
                            if det == "BCM1F": plt.ylim([100, 180])
                            elif det == "PLT": plt.ylim([250, 370])
                            elif det == "HFOC": plt.ylim([600, 1200])
                            elif det == "HFET": plt.ylim([2400, 4000])
                        i = 0
                        height = 120
                        if det == "HFOC": height = 700
                        if det == "HFET": height = 2700
                        for x in new_fill_pos_in_trains:
                            xpos = x -0.4
                            plt.text(xpos, height, f'FILL {fillcounter[i][1]}', horizontalalignment='left',verticalalignment='center', rotation='vertical')
                            i += 1
                        plt.savefig(f"{output_dir}/sigmavis_vs_train_leading_bunches_{det}_{fit}_{corr}_{energy}.pdf")
                        plt.clf()
                        del f

                        ########## plot of sigma vis of leading bunches (AVERAGED OVER EACH TRAIN) as a function of SCAN.
                        enumerate_scans = list(range(num_scans))
                        avg_sigmavis_of_scan_leading = []
                        f = plt.figure(figsize=(8,5))
                        ax = f.add_subplot(111)

                        plt.vlines(x=[x[0] - 0.5 for x in fillcounter], ymin=0, ymax=4000, colors=['purple'], ls='--', lw=1, alpha=0.5)
                        for i in range(num_scans):  # looping over each scan
                            avg_sigmavis_of_scan_leading.append(mean(traincounter_leading[i][1]))

                        plt.scatter(enumerate_scans,avg_sigmavis_of_scan_leading, marker=".", color='blue', alpha=1,s=100, zorder=4)

                        avg_sigma_vis_over_all_scans = mean(avg_sigmavis_of_scan_leading)
                        x = np.arange(0, num_scans, 1)
                        avg_sigma_vis_array = np.full_like(x, avg_sigma_vis_over_all_scans, dtype=np.double)
                        plt.plot(x, avg_sigma_vis_array, label=f'avg={round(avg_sigma_vis_over_all_scans, 2)}'r'$\mu b$', color='darkorange', zorder=1)
                        plt.fill_between(x, avg_sigma_vis_array + rmse(avg_sigma_vis_array, avg_sigmavis_of_scan_leading),
                                         avg_sigma_vis_array - rmse(avg_sigma_vis_array,  avg_sigmavis_of_scan_leading),
                                         alpha=0.5, label=f'rmse ={round(rmse(avg_sigma_vis_array,  avg_sigmavis_of_scan_leading), 2)}'r'$\mu b$',
                                         color='darkorange', zorder=2)

                        plt.legend()
                        plt.xlabel('Scan', size=13)
                        plt.ylabel(r'$\sigma_{vis} \,[\mu b]$', size=13)
                        plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
                        plt.text(0.20, 1.04, 'Preliminary', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
                        plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes, size=14)
                        plt.text(0.03, 0.95, f'Luminometer: {det}', ha='left', va='center', transform=ax.transAxes, size=13)
                        plt.text(0.03, 0.89, f'Fit function: {fit}', ha='left', va='center', transform=ax.transAxes, size=13)
                        plt.text(0.03, 0.83, f'Correction: {corr}', ha='left', va='center', transform=ax.transAxes,  size=13)
                        plt.text(0.03, 0.77, f'Average over LEADING BUNCHES in each scan', ha='left', va='center', weight='bold', transform=ax.transAxes, size=13)

                        plt.xlim([np.min(enumerate_scans) - 1, np.max(enumerate_scans) + 1])
                        if energy == "13600":
                            if det == "BCM1F":
                                plt.ylim([100, 180])
                            elif det == "PLT":
                                plt.ylim([250, 370])
                            elif det == "HFOC":
                                plt.ylim([600, 1200])
                            elif det == "HFET":
                                plt.ylim([2400, 4000])
                        i = 0
                        height = 120
                        if det == "HFOC": height = 700
                        if det == "HFET": height = 2700
                        for x in fillcounter:
                            xpos = x[0] - 0.4
                            plt.text(xpos, height, f'FILL {fillcounter[i][1]}', horizontalalignment='left', verticalalignment='center', rotation='vertical',size=8)
                            i += 1
                        plt.savefig(f"{output_dir}/sigmavis_vs_scan_leading_bunches_{det}_{fit}_{corr}_{energy}.pdf")
                        plt.clf()
                        del f

