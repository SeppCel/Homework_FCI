import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#loading files
csv_x1 = r"./pressure_8734.csv"
csv_x2 = r"./pressure_8606.csv"

df1 = pd.read_csv(csv_x1)
df2 = pd.read_csv(csv_x2)

# segnali di pressione
pressioni = df1["pressure_value"].values
x2 = df2["pressure_value"].values

#asse delle ascisse definito dalle ore a cui vengono lette le x1
ore = df1['hour'].str.split(':').str[0].astype(int).values

#funzioni utili
def rect(n):
    return np.where(np.abs(n) <= 0.5, 1, 0)

def tri(n):
    return np.where(np.abs(n) <= 1, 1 - np.abs(n), 0)

def valore_medio(sig):
    return np.mean(sig)

def energia(sig): #tempo discreto
    return np.sum(np.dot(sig, sig))

def sinc_filter(n):
    return np.sinc(n)



###grafici in una sola finestra###

#finestra unica
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

#GRAFICO 1---------------
energia_pressioni_1 = energia(pressioni)
axs[0, 0].plot(ore, pressioni, label='Funzione', color='#1f77b4', linewidth=1)
axs[0, 0].axhline(valore_medio(pressioni), linestyle='--', color='darkred', label=f'Valore Medio: {valore_medio(pressioni):.2f}')
axs[0, 0].set_xlim(min(ore), max(ore))
axs[0, 0].set_ylim(min(pressioni), max(pressioni))
axs[0, 0].set_xlabel("Ore")
axs[0, 0].set_ylabel("Pressioni")
axs[0, 0].set_title("Segnale Pressioni")
axs[0, 0].grid(True, linestyle=':', alpha=0.5)
axs[0, 0].legend()
axs[0, 0].text(0.70, 0.10, f'Energia = {energia_pressioni_1:.2f}', transform=axs[0,0].transAxes, 
         bbox=dict(facecolor='white', edgecolor='black'))



##ESERCIZIO 2##

N = len(pressioni)
n_indici = np.arange(N)
n_centered = n_indici - (N - 1) / 2 

B = 0.1 #Fattore scalatura sinc


h_x = B * np.sinc(B * n_centered)

#filtro normalizzato
h_x = h_x / np.sum(h_x)


y_n = np.convolve(pressioni, h_x, mode="same")


#Esercizio 2 punto b
x1N = pressioni - np.mean(pressioni)
y1N = y_n - np.mean(y_n) # y_n ora è quello corretto
r_xx = np.correlate(x1N, x1N, mode='full')
r_yy = np.correlate(y1N, y1N, mode='full')
lag = np.arange(-len(pressioni) + 1, len(pressioni))


#GRAFICO 2a

axs[0, 1].plot(ore, y_n, label='Segnale Filtrato y_n', color="#060002", linewidth=1.5)
axs[0, 1].set_title("Segnale filtrato (y_n) e segnale originale")
axs[0, 1].set_xlabel("Orario")
axs[0, 1].set_ylabel("Ampiezza")
axs[0, 1].grid(True, linestyle=':', alpha=0.5)

axs[0, 1].plot(ore, pressioni, label='Funzione Originale', color="#17C2D2", linewidth=0.5)
axs[0, 1].legend() 


axs[1, 0].plot(n_centered, h_x, label='Filtro Sinc(x/10)', color="#FF5733", linewidth=1)
axs[1, 0].set_title("Filtro Sinc Applicato (Corretto)")
axs[1, 0].set_xlabel("Campioni (n)")
axs[1, 0].set_ylabel("Ampiezza")
axs[1, 0].grid(True, linestyle=':', alpha=0.5)
axs[1, 0].legend()


##GRAFICO 2B
#Autocorrelazione X
axs[1, 1].plot(lag, r_xx, label='Autocorrelazione x', color='blue', linewidth=1)
axs[1, 1].set_title("Confronto Autocorrelazione r_xx e r_yy")
#aggiunto un box con l'energia r_xx[0] e r_yy[0], utile?
axs[1, 1].text(0.05, 0.97,
             f'Autocorrelazione x(0) = {r_xx[len(r_xx)//2]:.2f}',  
             transform=axs[1, 1].transAxes, 
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#Autocorrelazione Y
axs[1, 1].plot(lag,r_yy, label='Autocorrelazione y', color='orange', linewidth=1)
axs[1, 1].text(0.05, 0.90,
             f'Autocorrelazione y(0) = {r_yy[len(r_yy)//2]:.2f}',  
             transform=axs[1, 1].transAxes, 
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axs[1, 1].set_xlabel("Ritardo (Lag)")
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend()

correlazione_xy = np.corrcoef(pressioni, y_n)[0, 1]
testo_corr = f'Correlazione X e Y: {correlazione_xy:.4f}'
axs[1, 1].text(0.95, 0.80,
             testo_corr,  
             transform=axs[1, 1].transAxes, 
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


##CALCOLO VARIANZA E ENERGIA##

var_x = np.var(x1N)
var_y = np.var(y1N) 

print(f"--- Varianza ---")
print(f"Varianza segnale originale (x1N): {var_x:.2f}")
print(f"Varianza segnale filtrato (y1N): {var_y:.2f}")

energia_x = r_xx[N - 1] 
energia_y = r_yy[N - 1] 

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



#_______________________________________________________________________
##ESERCIZIO 3##

#punto a

x2N = x2 - np.mean(x2)            

if len(x1N) != len(x2N):
    min_length = min(len(x1N), len(x2N))
    x1N = x1N[:min_length]
    x2N = x2N[:min_length] 

delta_x = np.abs(x1N - x2N)

figes3, axs_es3 = plt.subplots(3,1, figsize=(15,10), sharex=True)
figes3.suptitle("Salto e Segnali di Pressione nei Nodi 8734 e 8606")

#nodo 8734
axs_es3[0].plot(ore, x1N, label='$x_{1N}[n]$ (Nodo 8734)', color='#1f77b4', linewidth=1)
axs_es3[0].set_title('Segnale $x_{1N}[n]$ (Nodo 8734)')
axs_es3[0].set_ylabel('Pressione (centrata)')
axs_es3[0].grid(True, linestyle=':', alpha=0.7)
axs_es3[0].legend(loc='upper right')

#nodo 8606
axs_es3[1].plot(ore, x2N, label='$x_{2N}[n]$ (Nodo 8606)', color='#ff7f0e', linewidth=1)
axs_es3[1].set_title('Segnale $x_{2N}[n]$ (Nodo 8606)')
axs_es3[1].set_ylabel('Pressione (centrata)')
axs_es3[1].grid(True, linestyle=':', alpha=0.7)
axs_es3[1].legend(loc='upper right')

#salto
axs_es3[2].plot(ore, delta_x, label='$\Delta x[n] = |x_{1N} - x_{2N}|$', color='#d62728', linewidth=1)
axs_es3[2].set_title('Salto di Pressione $\Delta x[n]$')
axs_es3[2].set_xlabel('Campione (n)', fontsize=12)
axs_es3[2].set_ylabel('Differenza Assoluta')
axs_es3[2].grid(True, linestyle=':', alpha=0.7)
axs_es3[2].legend(loc='upper right')

figes3.tight_layout(rect=[0, 0.03, 1, 0.95])


#punto b

K = 3

windows_x1N = np.array_split(x1N, K)
windows_x2N = np.array_split(x2N, K)

rho_k_list = [] 

print(f"Calcolo coefficienti di correlazione per K={K} finestre:")

for k in range(K):
    finestra_x1 = windows_x1N[k]
    finestra_x2 = windows_x2N[k]
    
    rho_k = np.corrcoef(finestra_x1, finestra_x2)[0, 1]
    
    rho_k_list.append(rho_k)
    
    print(f"  Finestra k={k+1}: rho = {rho_k:.4f}")

print(f"\nLista dei coefficienti di correlazione: {rho_k_list}")


fig.tight_layout(pad = 3.0)
plt.show()