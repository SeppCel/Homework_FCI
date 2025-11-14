import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#loading files
csv_x1 = r"./pressure_8734.csv"
csv_x2 = r"./pressure_8606.csv"

df1 = pd.read_csv(csv_x1)
df2 = pd.read_csv(csv_x2)

# segnali di pressione
x1 = df1["pressure_value"].values
x2 = df2["pressure_value"].values

# --- CODICE ---

p34 = open(csv_x1, "r") # RICORDA DI CHIUDERE IL FILE CON p34.close()
p34.readline()
p34.readline()

ore = list()
pressioni = list()
for riga in p34.readlines():
    r = riga.strip("\n").split(",")
    ore.append(int(r[0].split(":")[0]))
    pressioni.append(float(r[1]))

ore = np.array(ore)
pressioni = np.array(pressioni)

p34.close()


#funzioni utili
def rect(n):
    return np.where(np.abs(n) <= 0.5, 1, 0)

def tri(n):
    return np.where(np.abs(n) <= 1, 1 - np.abs(n), 0)

def valore_medio(sig):
    return np.mean(sig)

def energia(sig):
    return np.sum(np.dot(sig, sig))

def sinc_filter(n):
    return np.sin(n)/(n)



#Esercizio 2

x_spostato = np.subtract(ore, (len(ore)-1)/2)
h_x = np.sinc(x_spostato)

N = len(pressioni)
n = np.arange(N)
n_centered = n - (N - 1) / 2

B = 0.1

h_x_reale = np.sinc(B * n_centered)
h_x_reale = h_x_reale / np.sum(h_x_reale)

y_n = np.convolve(pressioni, h_x_reale, mode="same")



#Esercizio 2 punto b
#autocorrelazione segnali x e y

x1N = pressioni - np.mean(pressioni)
y1N = y_n - np.mean(y_n)
r_xx = np.correlate(x1N, x1N, mode='full')
r_yy = np.correlate(y1N, y1N, mode='full')
lag = np.arange(-len(pressioni) + 1, len(pressioni))


###grafici in una sola finestra###

#finestra unica
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

#GRAFICO 1---------------
energia = np.sum(pressioni**2)
axs[0, 0].plot(ore, pressioni, label='Funzione', color='#1f77b4', linewidth=1)
axs[0, 0].axhline(valore_medio(pressioni), linestyle='--', color='darkred', label=f'Valore Medio: {valore_medio(pressioni):.2f}')
axs[0, 0].set_xlim(min(ore), max(ore))
axs[0, 0].set_ylim(min(pressioni), max(pressioni))
axs[0, 0].set_xlabel("Orario")
axs[0, 0].set_ylabel("Pressioni")
axs[0, 0].set_title("Segnale Pressioni")
axs[0, 0].grid(True, linestyle=':', alpha=0.5)
axs[0, 0].legend()
axs[0, 0].text(0.70, 0.10, f'Energia = {energia:.2f}', transform=axs[0,0].transAxes, 
         bbox=dict(facecolor='white', edgecolor='black'))


#GRAFICO 2a---------------
axs[0, 1].plot(x_spostato, y_n, label='Segnale Filtrato y_n', color="#060002", linewidth=1)
axs[0, 1].set_ylim(31,38)
axs[0, 1].set_title("Segnale filtrato (y_n) e segnale originale")
axs[0, 1].set_xlabel("Campioni (spostati)")
axs[0, 1].set_ylabel("Ampiezza")
axs[0, 1].grid(True, linestyle=':', alpha=0.5)
axs[0, 1].legend()

axs[0, 1].plot(x_spostato, pressioni, label='Funzione Originale', color="#17C2D2", linewidth=0.5)
axs[0, 1].set_xlabel("Campioni (spostati)")
axs[0, 1].set_ylabel("Pressione")
axs[0, 1].grid(True, linestyle=':', alpha=0.5)
axs[0, 1].legend()

axs[1, 0].plot(h_x_reale, label='Filtro Sinc', color="#FF5733", linewidth=1)
axs[1, 0].set_title("Filtro Sinc Applicato")
axs[1, 0].set_xlabel("Campioni")
axs[1, 0].set_ylabel("Ampiezza")
axs[1, 0].grid(True, linestyle=':', alpha=0.5)
axs[1, 0].legend()


##GRAFICO 2B---------------
#Autocorrelazione X
axs[1, 1].plot(lag, r_xx, label='Autocorrelazione x', color='blue', linewidth=1)
axs[1, 1].set_title("Autocorrelazione pressioni (Energia)")
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend()
axs[1, 1].set_yticks([]) 
#Autocorrelazione Y
axs[1, 1].plot(lag,r_yy, label='Autocorrelazione y', color='orange', linewidth=1)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend()
axs[1, 1].set_yticks([]) 

correlazione_xy = np.corrcoef(pressioni, y_n)[0, 1]
testo_corr = f'Correlazione X e Y: {correlazione_xy:.4f}'
axs[1, 1].text(0.95, 0.80,
             testo_corr,  
             transform=axs[1, 1].transAxes, 
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


#x_corr = int(np.correlate(pressioni, pressioni))
#y_corr = int(np.correlate(y_n, y_n))
#testo_energia = f'Energia X: {x_corr}\nEnergia Y: {y_corr}'
#axs[1, 1].text(0.95, 0.95,
#             testo_energia,  
#             transform=axs[1, 1].transAxes, 
#             fontsize=10,
#             verticalalignment='top',
#             horizontalalignment='right',
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#axs[2, 1].axis('off')


##CALCOLO VARIANZA E ENERGIA##

var_x = np.var(x1N)
var_y = np.var(y1N)

print(f"--- Varianza ---")
print(f"Varianza segnale originale (x1N): {var_x:.2f}")
print(f"Varianza segnale filtrato (y1N): {var_y:.2f}")

N = len(x1N) 

energia_x = r_xx[N - 1] 
energia_y = r_yy[N - 1]

# energia_x = np.correlate(x1N, x1N, mode='valid')[0]
# energia_y = np.correlate(y1N, y1N, mode='valid')[0]

print(f"\n--- Energia (Autocorrelazione a Lag 0) ---")
print(f"Energia segnale originale (r_xx[0]): {energia_x:.2f}")
print(f"Energia segnale filtrato (r_yy[0]): {energia_y:.2f}")




half_max_x = energia_x / 2.0
half_max_y = energia_y / 2.0

width_x = np.sum(r_xx > half_max_x)
width_y = np.sum(r_yy > half_max_y)

print(f"\n--- Larghezza Lobo Centrale (a metà altezza) ---")
print(f"Larghezza lobo segnale originale (x1N): {width_x} campioni")
print(f"Larghezza lobo segnale filtrato (y1N): {width_y} campioni")




# Spaziatura e visualizzazione
plt.tight_layout(pad = 3.0)
plt.show()