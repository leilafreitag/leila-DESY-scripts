import pandas as pd, pylab as py, numpy as np, tables as t
import matplotlib.dates as mdates

# script to analyse the behavior of the empty bunches immediately after colliding bunches.
# folder_reprocessed is some reprocessed fill with the original channel mask and albedo correction
# folder_reprocessed_newChannels is the same fill reprocessed with the new channel mask and new albedo correction

#folder_reprocessed = '/localdata/lfreitag/22/7921/'
#folder_reprocessed = '/localdata/lfreitag/22/8088/'
folder_reprocessed = '/localdata/lfreitag/22/8094/'
folder_reprocessed = '/localdata/lfreitag/22/7966/'

folder_reprocessed_newChannels = '/localdata/lfreitag/newChannels/22/8088/'
folder_reprocessed_newChannels = '/localdata/lfreitag/newChannels/22/8094/'
folder_reprocessed_newChannels = '/localdata/lfreitag/newChannels/22/7966/'

#files = ['7921_355131_2207060104_2207060104.hd5','7921_355132_2207060104_2207060213.hd5','7921_355133_2207060213_2207060419.hd5','7921_355134_2207060419_2207060518.hd5','7921_355135_2207060518_2207061215.hd5']
#files = ['8088_356714_2208032319_2208032319.hd5','8088_356715_2208032319_2208032336.hd5','8088_356718_2208032336_2208040054.hd5','8088_356719_2208040054_2208040649.hd5','8088_356720_2208040649_2208040731.hd5','8088_356721_2208040731_2208040749.hd5','8088_356722_2208040749_2208041135.hd5']
#files = ['8094_356805_2208050040_2208050053.hd5','8094_356808_2208050053_2208050142.hd5','8094_356809_2208050142_2208050202.hd5','8094_356810_2208050202_2208050312.hd5','8094_356811_2208050312_2208050333.hd5','8094_356812_2208050333_2208050422.hd5','8094_356813_2208050422_2208050446.hd5','8094_356814_2208050446_2208050911.hd5','8094_356815_2208050911_2208051038.hd5','8094_356824_2208051038_2208051106.hd5','8094_356825_2208051106_2208051106.hd5']
files = ['7966_355429_2207111505_2207111505.hd5','7966_355430_2207111505_2207111505.hd5','7966_355431_2207111505_2207111636.hd5','7966_355434_2207111636_2207111702.hd5','7966_355435_2207111702_2207111740.hd5','7966_355437_2207111740_2207111747.hd5','7966_355438_2207111747_2207111754.hd5','7966_355439_2207111754_2207111801.hd5','7966_355440_2207111801_2207111805.hd5','7966_355441_2207111805_2207111812.hd5','7966_355442_2207111812_2207111822.hd5','7966_355443_2207111822_2207111957.hd5','7966_355444_2207111957_2207112100.hd5','7966_355445_2207112100_2207112234.hd5']


fill = 7966

time = []
bx_reprocessed = []
bx_reprocessed_newChannels =[]
bxraw_reprocessed_newChannels =[]
bxraw_reprocessed = []
ls_and_nb_reprocessed = []
num_rows_reprocessed = 0

for file in files:
	h5 = t.open_file(folder_reprocessed+file)
	for row in h5.root.bcm1flumi.iterrows():
		time.append(pd.to_datetime(row['timestampsec']*1000+row['timestampmsec'],unit='ms'))
		num_rows_reprocessed+=1
		bxraw_reprocessed.append(row['bxraw'])
		bx_reprocessed.append(row['bx'])
		ls_and_nb_reprocessed.append([row['lsnum'], row['nbnum']]) #so at each index get ls and nb
	h5.close()
	
for file in files:
	h5 = t.open_file(folder_reprocessed_newChannels+file)
	for row in h5.root.bcm1flumi.iterrows():
		bxraw_reprocessed_newChannels.append(row['bxraw'])
		bx_reprocessed_newChannels.append(row['bx'])
	h5.close()

bxraw_reprocessed_tot = [sum(x) for x in zip(*bxraw_reprocessed)]
bxraw_reprocessed_tot_newChannels = [sum(x) for x in zip(*bxraw_reprocessed_newChannels)]


print(bxraw_reprocessed_tot)


#print("time = ",time )
#print('ls_and_nb_reprocessed =',ls_and_nb_reprocessed)
#print('bxraw_reprocessed = ',bxraw_reprocessed)

# plotting the total rate measurement for each bcid
fig,ax = py.subplots(facecolor='white')
py.step([x+1 for x in range(3564)],bxraw_reprocessed_tot,where='mid', alpha = 0.5,label = 'reprocessed with albedo correction')
py.step([x+1 for x in range(3564)],bxraw_reprocessed_tot_newChannels,where='mid', alpha = 0.5,label = 'reprocessed with albedo correction and new channel mask')
py.xlabel('BCID')
py.ylabel('total bxraw')
py.title(f'total bxraw vs bcid (Fill {fill})')
py.legend()

# bcids of first and last colliding bunch in each train. got from makeplots_trains.py
# for filling scheme of 8088
train_starts = [198, 253, 308, 363, 442, 497, 552, 607, 792, 840, 895, 950, 1005, 1092, 1147, 1202, 1257, 1336, 1391, 1446, 1501, 1733, 1788, 1843, 1898, 1986, 2041, 2096, 2151, 2230, 2285, 2340, 2395, 2577, 2880, 2935, 2990, 3045, 3124, 3179, 3234, 3289]
train_ends = [245, 300, 355, 410, 489, 544, 599, 654, 792, 887, 942, 997, 1052, 1139, 1194, 1249, 1304, 1383, 1438, 1493, 1548, 1780, 1835, 1890, 1945, 2033, 2088, 2143, 2198, 2277, 2332, 2387, 2442, 2577, 2927, 2982, 3037, 3092, 3171, 3226, 3281, 3336]
# for filling scheme of 7966
train_starts = [21, 1170, 1629, 1806, 2064, 2523, 2958]
train_ends = [21, 1181, 1640, 1806, 2075, 2534, 2969]

num_trains = len(train_starts)
train_lengths = [e-s+1 for s,e in zip(train_starts,train_ends)]

# for each empty bunch that comes directly after a colliding bunch, plot its time evolution
first_empty_bunches = [x+1 for x in train_ends]

bxraw_of_one_bcid_reprocessed = [] #list of lists, each small list being the all the measurements in time for a given bcid
bxraw_of_one_bcid_reprocessed_newChannels = [] 
for elem in first_empty_bunches: #loop over the bcids of each first empty bunch
    temp = []
    temp_newChannels = []
    for i in range(len(ls_and_nb_reprocessed)): #go over all the measurements for this bcid in time
        temp.append(bxraw_reprocessed[i][elem-1])
        temp_newChannels.append(bxraw_reprocessed_newChannels[i][elem-1])

    bxraw_of_one_bcid_reprocessed.append(temp)
    bxraw_of_one_bcid_reprocessed_newChannels.append(temp_newChannels)

colors = ['red','orange','gold','yellowgreen','green','deepskyblue','blue','indigo']
colors = ['red','orange','gold','yellowgreen','green','aquamarine','deepskyblue','cornflowerblue','blue','slateblue','indigo','darkmagenta','magenta','red','orange','gold','yellowgreen','green','aquamarine','deepskyblue','cornflowerblue','blue','slateblue','indigo','darkmagenta','magenta','red','orange','gold','yellowgreen','green','aquamarine','deepskyblue','cornflowerblue','blue','slateblue','indigo','darkmagenta','magenta','red','orange','gold','yellowgreen','green','aquamarine','deepskyblue','cornflowerblue','blue','slateblue','indigo','darkmagenta','magenta','red','orange','gold','yellowgreen','green','aquamarine','deepskyblue','cornflowerblue','blue','slateblue','indigo','darkmagenta','magenta']

'''
# plots of bxraw vs time for individual bcids after colliding bunches
for i,elem in enumerate(first_empty_bunches):
    if i%4 == 0: # make new figure every four bcids
        fig, ax = py.subplots(facecolor='white',figsize=(8,5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # if want real time on x axis
        py.xlabel('time')
        py.ylabel('bxraw')
        py.ylim(-0.0006,0.00005)
        py.title(f'bxraw vs. time for empty bunches directly after trains (new channel mask)')
    #py.step(range(len(ls_and_nb_reprocessed)), bxraw_of_one_bcid_reprocessed[i], where='mid', label=f'bcid {elem}')
    ax.plot(pd.Series(bxraw_of_one_bcid_reprocessed_newChannels[i], index=time).rolling(1).mean(),color=colors[i],label=f'bcid {elem}')  # if want real time on x axis
    py.legend()
'''


# plots of bxraw tot for all first empty bcids, as a function of bcid
bxraw_first_empty_bunches = [sum(x) for x in bxraw_of_one_bcid_reprocessed]
bxraw_first_empty_bunches_newChannels = [sum(x) for x in bxraw_of_one_bcid_reprocessed_newChannels]
fig,ax = py.subplots(facecolor='white')
#py.plot(first_empty_bunches,bxraw_first_empty_bunches, alpha = 0.8)
py.scatter(first_empty_bunches,bxraw_first_empty_bunches, alpha = 0.8, label = 'original channel mask + albedo')
py.scatter(first_empty_bunches,bxraw_first_empty_bunches_newChannels, alpha = 0.8, label='new channel mask + albedo')
py.xlabel('BCID')
py.ylabel('total bxraw')
py.title(f'bxraw for first empty bunches after colliding bunches (Fill {fill})')
py.legend()

py.show()
