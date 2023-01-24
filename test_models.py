import numpy as np
import matplotlib.pyplot as plt


print ("loading...")
losses = np.load("last_losses_autoencoder_components_playing.npy",allow_pickle=True)

xs = np.linspace(0,len(losses)-1,len(losses))

dt_losses = []
inverse_dynamics_losses = []
forward_dynamics_losses = []
reconstruction_losses = []
klds = []
trex_losses = []

print ("traversing...")
for l in losses:
    dt_losses.append(l[0])
    inverse_dynamics_losses.append(l[1])
    forward_dynamics_losses.append(l[2])
    reconstruction_losses.append(l[3])
    klds.append(l[4])
    trex_losses.append(l[5])

print (len(dt_losses))
plt.figure(1)
plt.subplot(611)
plt.plot(xs, dt_losses)

plt.subplot(612)
plt.plot(xs, inverse_dynamics_losses)

plt.subplot(613)
plt.plot(xs, forward_dynamics_losses)

plt.subplot(614)
plt.plot(xs, reconstruction_losses)

plt.subplot(615)
plt.plot(xs, klds)

plt.subplot(616)
plt.plot(xs, trex_losses)


# reconstruction_loss_1 = []
# reconstruction_loss_2 = []
# bce1 = []
# bce2 = []
# kld1 = []
# kld2 = []

# print ("traversing...")
# for l in losses:
#     reconstruction_loss_1.append(l[0])
#     reconstruction_loss_2.append(l[1])
#     bce1.append(l[2])
#     bce2.append(l[3])
#     kld1.append(l[4])
#     kld2.append(l[4])

# print (len(reconstruction_loss_1))

# fig, ax = plt.subplots(2, 3)

# ax[0, 0].plot(xs, reconstruction_loss_1)
# ax[0, 1].plot(xs, bce1)
# ax[0, 2].plot(xs, kld1)

# ax[1, 0].plot(xs, reconstruction_loss_2)
# ax[1, 1].plot(xs, bce2)
# ax[1, 2].plot(xs, kld2)



plt.savefig('last_losses_autoencoder_components_playing.png')





