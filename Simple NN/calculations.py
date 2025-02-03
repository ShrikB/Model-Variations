import numpy as np

i1, i2 = [0.9, 0.3]
o1_target, o2_target = [0.01, 0.99]
w1, w2, w3, w4, w5, w6, w7, w8 = [0.8, -0.4, -0.2, 0.3, -0.7, -0.3, 0.5, 0.6]
b11, b12, b21, b22 = [0.1, 0.6, 0.2, 0.4]

net_h1 = i1*w1 + i2*w2 + 1*b11
net_h2 = i1*w3 + i2*w4 + 1*b12
out_h1 = 1/(1 + np.exp(-1*net_h1))
out_h2 = 1/(1 + np.exp(-1*net_h2))

net_o1 = out_h1*w5 + out_h2*w6 + 1*b21
net_o2 = out_h1*w7 + out_h2*w8 + 1*b22
out_o1 = 1/(1 + np.exp(-1*net_o1))
out_o2 = 1/(1 + np.exp(-1*net_o2))

err_o1 = 0.5*np.power((o1_target - out_o1), 2)
err_o2 = 0.5*np.power((o2_target - out_o2), 2)
err_tot = err_o1 + err_o2       #answer to question 1

###Calculating pd_eTot with respect to w5
#pd_eTot by out_o1
pd_etot_out_o1 = -(o1_target - out_o1)
pd_out_o1_net_o1 = out_o1*(1.0 - out_o1)
pd_net_o1_w5 = out_h1

pd_eTot_w5 = pd_etot_out_o1 * pd_out_o1_net_o1 * pd_net_o1_w5
#print(pd_eTot_w5)
###Calculating pd_eTot with respect to w6
#pd_etot_o1 = -(o1_target - out_o1)
#pd_out1_net1 = out_o1*(1.0 - out_o1)
pd_net_o2_w6 = out_h2

pd_eTot_w6 = pd_etot_out_o1 * pd_out_o1_net_o1 * pd_net_o2_w6
#print(pd_eTot_w6)
###Calculating pd_eTot with respect to w7
pd_etot_o2 = -(o2_target - out_o2)
pd_out2_net2 = out_o2*(1.0 - out_o2)
pd_net1_w7 = out_h1

pd_eTot_w7 = pd_etot_o2 * pd_out2_net2 * pd_net1_w7
#print(pd_eTot_w7)

###Calculating pd_eTot with respect to w8
#pd_etot_o2 = -(o2_target - out_o2)
#pd_out2_net1 = out_o2*(1.0 - out_o2)
pd_net2_w8 = out_h2

pd_eTot_w8 = pd_etot_o2 * pd_out2_net2 * pd_net2_w8
#print(pd_eTot_w8)

#w5, w6, w7, w8 = [(w5 - 0.5*pd_eTot_w5), (w6 - 0.5*pd_eTot_w6), (w7 - 0.5*pd_eTot_w7), (w8 - 0.5*pd_eTot_w8)] uncomment after propagation finished

###Calculating pd_eTot with respect to w1
pd_eo1_neto1 = pd_etot_out_o1 * pd_out_o1_net_o1
pd_neto1_outh1 = w5
pd_eo1_outh1 = pd_eo1_neto1 * pd_neto1_outh1

pd_eo2_neto2 = pd_etot_o2 * pd_out2_net2

pd_eo2_outh1 = 0