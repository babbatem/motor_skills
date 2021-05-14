import numpy as np
import matplotlib.pyplot as plt

train_sizes = []
test_accs = []
errs = []

ta = [0.5372233400402414, 0.5231388329979879, 0.5090543259557344, 0.5533199195171026, 0.5835010060362174]
train_sizes.append(3)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.4104627766599598, 0.5895372233400402, 0.5995975855130785, 0.5311871227364185, 0.44668008048289737]
train_sizes.append(6)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.5714285714285714, 0.6277665995975855, 0.607645875251509, 0.5331991951710262, 0.5211267605633803]
train_sizes.append(12)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.5231388329979879, 0.5694164989939637, 0.5191146881287726, 0.6156941649899397, 0.5110663983903421]
train_sizes.append(24)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.5150905432595574, 0.5653923541247485, 0.613682092555332, 0.5915492957746479, 0.6438631790744467]
train_sizes.append(49)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.5835010060362174, 0.5875251509054326, 0.6559356136820925, 0.5995975855130785, 0.6519114688128773]
train_sizes.append(99)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.6740442655935613, 0.613682092555332, 0.7082494969818913, 0.7565392354124748, 0.6317907444668008]
train_sizes.append(198)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.6861167002012073, 0.670020120724346, 0.670020120724346, 0.6458752515090543, 0.6579476861167002]
train_sizes.append(397)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.7424547283702213, 0.7303822937625755, 0.7283702213279678, 0.6780684104627767, 0.7585513078470825]
train_sizes.append(794)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.7585513078470825, 0.7424547283702213, 0.7082494969818913, 0.7203219315895373, 0.7243460764587525]
train_sizes.append(1192)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.7464788732394366, 0.7364185110663984, 0.7484909456740443, 0.7484909456740443, 0.7243460764587525]
train_sizes.append(1589)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

ta = [0.7525150905432596, 0.7424547283702213, 0.7364185110663984, 0.772635814889336, 0.7565392354124748]
train_sizes.append(1987)
test_accs.append(sum(ta)/len(ta))
errs.append(np.std(ta))

################################################################################################################

ae_train_sizes = []
ae_test_accs = []
ae_errs = []

ta = [0.613682092555332, 0.5271629778672032, 0.5432595573440644, 0.6398390342052314, 0.5050301810865191, 0.5653923541247485]
ae_train_sizes.append(3)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.5835010060362174, 0.4869215291750503, 0.6016096579476862, 0.5714285714285714, 0.5191146881287726]
ae_train_sizes.append(6)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.6116700201207244, 0.5533199195171026, 0.635814889336016, 0.482897384305835, 0.579476861167002]
ae_train_sizes.append(12)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.5513078470824949, 0.5975855130784709, 0.5352112676056338, 0.6217303822937625, 0.5694164989939637]
ae_train_sizes.append(24)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.5995975855130785, 0.5935613682092555, 0.6317907444668008, 0.6338028169014085, 0.5593561368209256]
ae_train_sizes.append(49)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.6438631790744467, 0.613682092555332, 0.6237424547283702, 0.6237424547283702, 0.5774647887323944]
ae_train_sizes.append(99)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.6378269617706237, 0.5754527162977867, 0.6237424547283702, 0.6156941649899397, 0.5754527162977867]
ae_train_sizes.append(198)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.6297786720321932, 0.6800804828973843, 0.6559356136820925, 0.6579476861167002, 0.6277665995975855]
ae_train_sizes.append(397)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.6378269617706237, 0.6740442655935613, 0.6297786720321932, 0.6961770623742455, 0.6438631790744467]
ae_train_sizes.append(794)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.6559356136820925, 0.6861167002012073, 0.6619718309859155, 0.6277665995975855, 0.6579476861167002]
ae_train_sizes.append(1192)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.6740442655935613, 0.676056338028169, 0.6659959758551308, 0.6438631790744467, 0.6217303822937625]
ae_train_sizes.append(1589)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

ta = [0.7022132796780685, 0.710261569416499, 0.6780684104627767, 0.6720321931589537, 0.7223340040241448]
ae_train_sizes.append(1987)
ae_test_accs.append(sum(ta)/len(ta))
ae_errs.append(np.std(ta))

################################################################################################################

plt.errorbar(train_sizes[:6], test_accs[:6], yerr=errs[:6], label="Pose")
plt.errorbar(ae_train_sizes[:6], ae_test_accs[:6], yerr=ae_errs[:6], label="3D Autoencoder")
plt.savefig("/home/mcorsaro/Desktop/plot_err_cut.jpg")
plt.legend(loc='lower right')
