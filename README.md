# scan-angle
please ONLY download the get_scan_angle notebook and the norm_xyz.npy file.
There is a simply function in the notebook. Input the ra, dec, epoch, and directory of norm_xyz.npy file (not including the name of the file), and get the ra/dec components (tan(theta) and cot(theta)) of the scan direction. 

Note: This function cannot distinguish whether it is along-scanning or anti-scanning direction. It can be the case that for some epochss it extracted along-scanning direction but for others anti-scanning direction. (Or at least I haven't think of this problem seriously. But the influence of scanning direction usually does not care whether it is along- or anti-scanning. )
