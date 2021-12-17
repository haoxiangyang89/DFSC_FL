# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:38:36 2018

@author: Thomas Massion
"""
import urllib
import csv

def make_csv(url,csvname):
    ''' makes csv from water data at url'''
    # leng = len(links)
    #for i in range(0,len(links)-1):
    f = urllib.request.urlopen(url)
    flist = f.readlines()
    #flist = flist[38:]
    #newnew = str(flist[12]).strip('b').split("\t")
    print(len(flist))
    with open(csvname,"w",newline="\n") as fw:
        fwrite = csv.writer(fw,delimiter=",")
        for line in flist:
            newline = str(line).split("\\t")
            fwrite.writerow(newline)
            new = str(line).split("\\t")
            print(new)
    return;

name1 = "miamidade_watdat.csv"
name2 = "keywest_watdat.csv"
link1 = "https://waterdata.usgs.gov/nwis/dv?cb_00010=on&cb_00065=on&cb_00095=on&cb_00480=on&cb_63158=on&format=rdb&site_no=02286328&referred_module=sw&period=&begin_date=2017-08-30&end_date=2017-09-13"
link2 = "https://waterdata.usgs.gov/nwis/dv?cb_00010=on&cb_00065=on&cb_00095=on&cb_00480=on&cb_63680=on&cb_72137=on&format=rdb&site_no=250802081035500&referred_module=sw&period=&begin_date=2017-08-30&end_date=2017-09-13"   
links = [link1,link2]
print(links[0])
print(links[1])
make_csv(link1,name1)
make_csv(link2,name2)




    #    for i in range(1,len(flist)):
    #        fw.writer(str(flist[i]))
    #        fw.write("\n")
    
    



#
#for i in range(0,len(myfile)-1):
#    flist[i] = str(myfile[i])
#    
#oline = 2637
#f = str(myfile[oline:])
#fs = f.split("\n")
#with open("waterdat.txt")
#webf.write(myfile)
#    #print(myfile)
#

#   # with open("waterdat.txt","r") as f:
#            f_contents = f.readline()
#    print(f_contents)
#    
#    
#f = open("waterdat.txt")
#for l in f.readlines():
#  print(l.strip().split("\t"))
#  break
#  f.close()
  
#with open("waterdat.txt","r",newline="\n") as fread:
#    freader = csv.reader(fread,delimiter="\t")
#    with open("waterdat.csv","w") as fwrite:
#        fwriter = csv.writer(fwrite,delimiter=',')
#        for row in freader:
#            print(row)
#            fwriter.writerow(row)
#fcut = myfile[2597:]
#with open("waterdat.txt","w",newline="\n") as webf:
#with open('waterdat.csv','wb') as webf:
 #   webread = csv.writer(webf,delimiter='\t')
    #print(myfile)
  #  end = len(myfile)
    #end = len(myfile) + end
   # oline = 2635
    #print(myfile[oline:])
    #counter = 0

#    while oline < len(myfile):
#        nline = myfile[oline:].find("\n")
#        nline = len(myfile[oline:]) + nline
#        webread.writerow(myfile[oline:(oline + nline)])
#    #    webf.write("\n")
#        print(myfile[oline:(oline + nline)])
#        oline = oline+nline+1
#        counter += 1

#webf.write(myfile[2595:])
#webf.write(myfile[])

#with open('waterdat.csv','wb') as webf:
#    webwrite = csv.writer(webf,delimiter='\t')
#    for row in myfile:
#        webwrite.writerow(myfile[row])
        