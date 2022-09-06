import pandas as pd
import numpy as np
import shutil
import os
import argparse
import matplotlib.pyplot as plt
from statistics import mean
import itertools

# makes plot of (sigmavis_vs_scan_scaled_comparison_detectors) or (sigmavis_vs_scan_scaled_comparison_fits)
# depending on if you input multiple detectors OR multiple fits to compare.

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('output', nargs='?', help='Output directory to store plots.')
    #args = parser.parse_args()

    output_dir = '/Users/leila/PycharmProjects/DESY/VDM' #args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # input either multiple dets OR multiple fits
    dets = ["BCM1F", "HFOC", "HFET", "PLT"] # "BCM1F", "HFOC", "HFET", "PLT"]
    fits = ["DG"] # "SGConst", "DG", "SG"]
    corrs = ["Background"] #"Background", "noCorr"
    energy = "13600"
    energy_str = "13.6 TeV" if energy == "13600" else "900 GeV"

    wantSigVisVsScanComparison = True

    #fills_13TeV = [7966,7978,8007, 8016, 8027,8058,8063,8067,8068,8072,8078,8079,8081,8087,8088,8091,8094,8102,8103,8106,8112,8113,8115,8118,8120,8124,8128] #7920, 7921,7960,7963,
    fills_13TeV = [7966,7978,8007, 8016, 8027,8058,8063,8067,8068,8072,8078,8079,8081,8087,8088,8091,8102,8103,8106,8112,8113,8115,8118,8120,8124,8128] #7920, 7921,7960,7963,
    blacklist_fill = []

    def rmse(theoretical, data):
        return np.sqrt(mean((theoretical - data) ** 2))

    df = pd.read_csv("/Users/leila/PycharmProjects/DESY/VDM/fit_and_lumidata_averageBCIDs.csv", sep=',') #this file contains averages over the BCIDs for each scan.
    all_combinations = []
    check = [] #check will be a list of all the fillcounters(pos of new fill, fill number) for all the det fit corrs. to make sure they are the same.
    for det in dets:
        for corr in corrs:
            for fit in fits:
                print("---------------------------")
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
                fillcounter_for_check = [] #want the same as fillcounter but with one more entry in each fill to give info about scans in the last fill.
                last_fill_index = len(fills)-1
                for index, fill in enumerate(fills):
                    if "HF" in det and fill == 7733:
                        continue
                    df_samefill = partial_df[partial_df["fill"] == fill]
                    if np.isnan(np.mean(df_samefill["avg_xsec"])): # or np.isnan(np.sqrt(df_samefill["SBILErr"].pow(2).sum())):
                        continue

                    fillcounter.append([len(sigma_vis),fill])  # the length of sigma vis should give me the index the fill starts at, and what fill it is
                    fillcounter_for_check.append([len(sigma_vis),fill])
                    #print(len(sigma_vis),fill)

                    sigma_vis += df_samefill["avg_xsec"].tolist()
                    sbil += df_samefill["avg_SBIL"].tolist()
                    sigma_vis_err += df_samefill["avg_xsecErr"].tolist()
                    #sbil_err += df_samefill["SBILErr"].tolist()

                    if index == last_fill_index: fillcounter_for_check.append([len(sigma_vis),fill]) #solves the problem for the combination not having the same scans in last fill

                    for i in range(len(df_samefill["avg_xsec"].tolist())):
                        #eff_fills.append(fill-i*2)
                        eff_fills.append(fill+0.1*i) #should be pos?
                check.append(fillcounter_for_check)

                print("Effective fills: ", eff_fills)
                print("SBIL: ", sbil)
                print("sigmavis: ", sigma_vis)
                all_combinations.append([det,fit,corr,eff_fills,sbil,sigma_vis,sigma_vis_err])


    # finding if all combos of det fit corr has same scans and can be compared.

    checklist = [list(group) for key, group, in itertools.groupby(check)]
    if len(checklist) != 1:
        wantSigVisVsScanComparison=False
        print("=====================================================================\n"
              "VERY BAD. MAKE SURE EACH DET FIT CORR COMBINATION HAS THE SAME SCANS\n"
              "=====================================================================")
        big_l = []
        for item in itertools.groupby(check):
            l = list(item)
            l.pop()
            tuples = [] #want to change l[0] into list of tuples instead of list of lists so can use set later
            for x in l[0]:
                tuples.append(tuple(x))
            big_l.append(tuples)
        difference_total = []
        for i in range(len(big_l)-1):#i is indices of lists within big l
            difference = set(big_l[i]).symmetric_difference(set(big_l[i+1]))
            difference_total += difference #NEED TO UPDATE THIS TO GET RID OF DUPLICATED FILL NUMBERS THAT ARE APPENDED
        print("THESE FILLS ARE NOT UNIVERSAL (enumerate scan index, fill number):", difference_total)

    if wantSigVisVsScanComparison:
        # plot of sigvis vs scan but normalised, to compare for diff luminometers/fits
        f = plt.figure(figsize=(8, 5))
        ax = f.add_subplot(111)
        plt.vlines(x=[x[0] - 0.5 for x in fillcounter], ymin=0, ymax=4000, colors=['purple'],
                   ls='--', lw=1, alpha=0.5)

        combination_number = 0
        colors = ['red','orange','gold','yellowgreen','green','deepskyblue','blue','indigo']
        for combination in all_combinations:
            det = combination[0]
            fit = combination[1]
            corr = combination[2]
            eff_fills = combination[3]
            sbil = combination[4]
            sigma_vis = combination[5]
            sigma_vis_err = combination[6]
            num_scans = len(eff_fills)
            enumerate_scans = list(range(num_scans))
            avg_sigma_vis = mean(sigma_vis)
            #x = np.arange(0, num_scans, 1)
            # plt.fill_between(x, avg_sigma_vis_array + rmse(avg_sigma_vis_array, sigma_vis),
            #                 avg_sigma_vis_array - rmse(avg_sigma_vis_array, sigma_vis),
            #                 alpha=0.5, label=f'rmse ={round(rmse(avg_sigma_vis_array, sigma_vis), 2)}',
            #                 color='darkorange', zorder=2)

            sigma_vis_scaled = [val / avg_sigma_vis for val in sigma_vis]
            sigma_vis_err_scaled = [val / avg_sigma_vis for val in sigma_vis_err]
            plt.scatter(enumerate_scans, sigma_vis_scaled,  marker=".",alpha=0.8, s=100, zorder=4, color = colors[combination_number], label = f'{det} {fit} {corr}')
            plt.errorbar(enumerate_scans, sigma_vis_scaled, yerr=sigma_vis_err_scaled, alpha=0.8, ls="none", color=colors[combination_number],
                         zorder=3)
            combination_number+=1

        #avg_sigma_vis_array = np.full_like(x, 1, dtype=np.double)
        #plt.plot(x, avg_sigma_vis_array, label=r'$\overline{\sigma_{vis}}$'f'={round(avg_sigma_vis, 2)}',color='darkorange', zorder=1)  # plotting the line of average sigvis

        plt.legend()
        plt.xlabel('Scan', size=13)
        plt.ylabel(r'$\sigma_{vis} \, / \, \overline{\sigma_{vis}}$', size=13)
        plt.text(0.06, 1.04, 'CMS', ha='center', va='center', transform=ax.transAxes, weight="bold",size=14)
        plt.text(0.20, 1.04, 'Preliminary', ha='center', va='center', transform=ax.transAxes,style='italic', size=13)
        plt.text(0.87, 1.04, r'($\sqrt{s}$='f'{energy_str})', ha='center', va='center', transform=ax.transAxes, size=14)
        plt.text(0.03, 0.95, f'Average over BCIDs for each scan', ha='left', va='center',weight='bold',transform=ax.transAxes, size=13)
        plt.text(0.03, 0.89, f'Sigma visible scaled using', ha='left', va='center', transform=ax.transAxes,size=13)
        plt.text(0.03, 0.83, f'average per detector/fit/correction', ha='left', va='center', transform=ax.transAxes, size=13)
        plt.xlim([np.min(enumerate_scans) - 1, np.max(enumerate_scans) + 1])
        plt.ylim([0.7, 1.2])
        i = 0
        height = 0.8
        for x in fillcounter:
            xpos = x[0] - 0.5
            plt.text(xpos, height, f'FILL {fillcounter[i][1]}', horizontalalignment='left',verticalalignment='center', rotation='vertical')
            i += 1

        if len(dets)==1: plt.savefig(f"{output_dir}/sigmavis_vs_scan_scaled_comparison_fits_{det}.pdf")
        elif len(fits)==1: plt.savefig(f"{output_dir}/sigmavis_vs_scan_scaled_comparison_detectors_{fit}_{corr}.pdf")
        else: print("==========================================================================\n CHOOSE A COMPARISON TYPE BY ONLY USING A SINGLE DETECTOR OR FIT TYPE")
        plt.clf()
        del f
