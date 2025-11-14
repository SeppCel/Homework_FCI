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

def convoluzione(sig1, sig2):
    ris = np.zeros(len(sig1))
    for i in range(len(sig1)):
        print("sig1 " + str(sig1[i]),"sig2 " + str(sig2[-i-1]))
        ris[i] = sig1[i] * sig2[-i-1]
    return ris

print(convoluzione(np.arange(3),np.arange(3)))



# --- GRAFICO 1 ---
plt.figure(figsize=(15, 8))
plt.plot(ore, pressioni, label='funzione', color='#1f77b4', linewidth=0.5)
#plt.axvline(min(pressioni), linestyle='--', color='darkred', label=f'Valore Minimo: x={min(pressioni):.2f}')
plt.axhline(valore_medio(pressioni), linestyle='--', color='darkred', label=f'Valore Medio: {valore_medio(pressioni):.2f}')
plt.axhline(energia(pressioni), linestyle='--', color='darkred', label=f'Energia: {energia(pressioni):.2f}')
# Aggiunta di dettagli
plt.xlim(min(ore), max(ore))
plt.ylim(min(pressioni), max(pressioni))
plt.xlabel("Orario")
plt.ylabel("Pressioni")
plt.title("Pressioni durante il tempo")
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.show()



#Esercizio 2
x_spostato = np.subtract(ore, (len(ore)-1)/2)
h_x = np.sinc(x_spostato)

y_n = np.convolve(h_x, pressioni, mode="same")
y_n = convoluzione(h_x, pressioni)
print(y_n)
#grafico sinc centrata nel centro delle ore tempo discreto
plt.figure(figsize=(30, 10))

plt.subplot(2,3,1)
plt.plot(x_spostato, y_n, label='sinc', color='#1f77b4', linewidth=0.5)
plt.ylim(31,38)

#grafico originale pressioni
plt.subplot(2,3,2)
plt.plot(x_spostato, pressioni, label='funzione originale', color="#1AECFF", linewidth=0.5)

#grafico convoluzione pressioni e sinc
"""plt.subplot(2,3,3)
plt.plot(rect(1), y_n, label='convoluzione', color="#ff0606", linewidth=0.5)
plt.show()"""




#Esercizio 2 punto b


#autocorrelazione segnali x e y
x_corr = int(np.correlate(pressioni, pressioni))
y_corr = int(np.correlate(y_n, y_n))

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.axhline(x_corr, linestyle='--', color='darkred', label=f'autocorrelazione x: {x_corr:.2f}')
#plt.annotate(f'autocorrelazione x {x_corr:.2f}', xy=(0.2,820000))
plt.title("autocorrelazione pressioni")
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.axhline(y_corr, linestyle='--', color='darkred', label=f'autocorrelazione x: {y_corr:.2f}')
plt.title("autocrrelazione filtro")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


