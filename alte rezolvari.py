import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hic  # For HCA
import sklearn.decomposition as dec  # For PCA
import sklearn.cross_decomposition as sk  # For CCA
import sns
from scipy.cluster._hierarchy import linkage
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from learn.plots import dendrogram

# region SUBIECT DIVERSITATEA POPULATIEI CU ANALIZA FACTORIALA

output_dir = './data_out'

# ==========================================
# A. CERINTE LEGATE DE DIVERSITATE
# ==========================================

# --- A.1: Localități cu diversitate 0 ---
df_div = pd.read_csv('Diversitate.csv')
ani = [str(year) for year in range(2008, 2022)]

# Filtrare localități cu diversitate 0 în cel puțin un an
mask = (df_div[ani] == 0).any(axis=1)
df_cerinta1 = df_div[mask]
df_cerinta1.to_csv(f'{output_dir}/Cerinta1.csv', index=False)

# --- A.2: Maxim pe județe ---
df_coduri = pd.read_csv('Coduri_Localitati.csv')

# Unire tabele pentru a obține județul
merged_df = pd.merge(df_div, df_coduri[['Siruta', 'Judet']], on='Siruta', how='inner')

# Calculare medie pe anii 2008-2021
merged_df['Diversity_Mean'] = merged_df[ani].mean(axis=1)

# Identificare localitate cu media maximă pe fiecare județ
idx_max = merged_df.groupby('Judet')['Diversity_Mean'].idxmax()
result_df = merged_df.loc[idx_max, ['Judet', 'Localitate', 'Diversity_Mean']]

# Sortare alfabetică după județ
result_df = result_df.sort_values(by='Judet')
result_df.to_csv(f'{output_dir}/Cerinta2.csv', index=False)

# ==========================================
# B. ANALIZA FACTORIALA (DIVERSITATE)
# ==========================================

# --- Preprocesare comună pentru B ---
data = df_div[ani]
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# Analiză Factorială comună (2 factori, Varimax)
n_factors = 2
fa = FactorAnalysis(n_components=n_factors, rotation='varimax')
fa.fit(data_std)

# Obținere încărcări factoriale (loadings) - Matrice (n_variabile x n_factori)
loadings = fa.components_.T

# --- B.1: Varianța factorilor ---
varianta_factori = np.sum(loadings ** 2, axis=0)
total_varianta = len(ani)  # La date standardizate, varianța totală = nr. variabile
procent_varianta = (varianta_factori / total_varianta) * 100
procent_cumulat = np.cumsum(procent_varianta)

df_varianta = pd.DataFrame({
    'Varianta': varianta_factori,
    'Procent varianta': procent_varianta,
    'Procent cumulat': procent_cumulat
}, index=[f'Factor {i + 1}' for i in range(n_factors)])

df_varianta.to_csv(f'{output_dir}/Varianta.csv')

# --- B.2: Corelațiile factoriale ---
df_r = pd.DataFrame(loadings, index=ani, columns=[f'Factor {i + 1}' for i in range(n_factors)])
df_r.to_csv(f'{output_dir}/r.csv')

# --- B.3: Cercul corelațiilor ---
plt.figure(figsize=(8, 8))
plt.title('Cercul corelatiilor (Factor 1 vs Factor 2)')
plt.xlabel(f'Factor 1 ({procent_varianta[0]:.2f}%)')
plt.ylabel(f'Factor 2 ({procent_varianta[1]:.2f}%)')

# Desenare cerc unitar
cerc = plt.Circle((0, 0), 1, color='blue', fill=False, label='Corelație maximă (1)')
plt.gca().add_artist(cerc)

# Desenare vectori variabile
for i, an in enumerate(ani):
    x = loadings[i, 0]
    y = loadings[i, 1]
    # Săgeată de la origine la coordonatele încărcării
    plt.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc='red', ec='red', length_includes_head=True)
    # Etichetă an, cu un mic offset pentru lizibilitate
    plt.text(x, y, an, color='black', ha='center', va='bottom', fontsize=9)

# Setări axe și grilă
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid(True, linestyle='--')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()

# ==========================================
# C. ANALIZA COMPONENTELOR PRINCIPALE (vectori proprii)
# ==========================================
# 1. Încărcare matrice vectori proprii
df_a = pd.read_csv('a.csv')

# 2. Definire valori proprii (date în enunț)
eigenvalues = np.array([3.019, 1.2203, 0.6536, 0.102, 0.005])

# 3. Selectare date pentru primele 2 componente
# Vectorii proprii sunt pe coloane (primele 2 coloane)
vectors_cp1_cp2 = df_a.iloc[:, :2].values
eigenvalues_cp1_cp2 = eigenvalues[:2]

# 4. Calcul comunalități
# Comunalitatea = Suma(Corelatie^2) = Suma(VectorPropriu^2 * ValoareProprie)
communalities = np.sum(vectors_cp1_cp2 ** 2 * eigenvalues_cp1_cp2, axis=1)

# 5. Identificare și afișare variabile (> 90%)
variable_names = [f"X{i + 1}" for i in range(len(communalities))]

print("Variabilele observate cu comunalitate > 90%:")
found_any = False
for name, comm in zip(variable_names, communalities):
    if comm > 0.9:
        print(f"{name} (Comunalitate: {comm:.4f})")
        found_any = True

if not found_any:
    print("Nu s-au găsit variabile care să îndeplinească criteriul.")
# endregion

# region SUBIECT DIVERSITATEA POPULATIEI CU ANALIZA CLUSTER SI WARD

# Creare director output dacă nu există
output_dir = './data_out'

df = pd.read_csv('Diversitate.csv')
ani = [str(year) for year in range(2008, 2022)]

# ==========================================
# 1. Medie indice diversitate (Cerința anterioară)
# ==========================================
# Calcul medie pe rând (axis=1)
df['Indice_Mediu'] = df[ani].mean(axis=1)

# Sortare descrescătoare
df_sorted = df.sort_values(by='Indice_Mediu', ascending=False)

# Salvare rezultat 1
rezultat1 = df_sorted[['Siruta', 'Localitate', 'Indice_Mediu']]
rezultat1.to_csv(f'{output_dir}/Cerinta1.csv', index=False)

# ==========================================
# 2. Număr localități cu diversitate 0 pe județe
# ==========================================
df_coduri = pd.read_csv('Coduri_Localitati.csv')

# Unire tabele (merge) pe baza codului Siruta
merged_df = pd.merge(df, df_coduri[['Siruta', 'Judet']], on='Siruta', how='inner')

# Creăm o mască booleană: True unde diversitatea este 0, False în rest
# Selectăm doar coloanele anilor pentru această verificare
is_zero = merged_df[ani] == 0

# Adăugăm coloana Judet înapoi în acest dataframe de verificare pentru a putea grupa
is_zero['Judet'] = merged_df['Judet']

rezultat2 = is_zero.groupby('Judet')[ani].sum().reset_index()

# Salvare rezultat 2
rezultat2.to_csv(f'{output_dir}/Cerinta2.csv', index=False)

# ==========================================
# SECTIUNEA B: Analiza de Clusteri (Metoda Ward)
# ==========================================

# --- B.1 Matricea Ierarhie ---
X = df[ani].values
Z = linkage(X, method='ward')

linkage_df = pd.DataFrame(Z, columns=['Cluster 1', 'Cluster 2', 'Distanta', 'Nr_Instante'])
linkage_df['Cluster 1'] = linkage_df['Cluster 1'].astype(int)
linkage_df['Cluster 2'] = linkage_df['Cluster 2'].astype(int)
linkage_df['Nr_Instante'] = linkage_df['Nr_Instante'].astype(int)
print(linkage_df)

# --- B.2 Graficul Dendrogramă ---
plt.figure(figsize=(12, 7))
plt.title('Dendrograma - Partiția Optimală (Metoda Ward)')
plt.xlabel('Localități (Index)')
plt.ylabel('Distanță (Euclidiană)')

# Calculăm pragul pentru colorare și tăiere (consistent cu cerința de "partiție optimală" vizuală)
# De regulă, pragul default la dendrogramă este 70% din distanța maximă
threshold = np.max(Z[:, 2]) * 0.7

dendrogram(
    Z,
    no_labels=True,
    leaf_rotation=90.,
    leaf_font_size=8.,
    color_threshold=threshold
)
plt.axhline(y=threshold, c='black', lw=1, linestyle='dashed', label='Prag partiție')
plt.legend()
plt.show()

# --- B.3 Componența Partiției Optimale ---

# Folosim fcluster pentru a obține etichetele clusterilor pe baza pragului
# criterion='distance' taie arborele la distanța specificată (t=threshold)
labels = hic.fcluster(Z, t=threshold, criterion='distance')

# Adăugăm etichetele în DataFrame-ul original (ordinea se păstrează)
df['Cluster'] = labels

# Selectăm coloanele cerute
result_b3 = df[['Siruta', 'Localitate', 'Cluster']]

# Salvăm
result_b3.to_csv(f'{output_dir}/popt.csv', index=False)

# ==========================================
# SECTIUNEA C: Varianța specifică maximă (g20.csv)
# ==========================================

# Citim fișierul cu loading-uri (coeficienți factoriali)
# Presupunem că fișierul conține doar valorile numerice sau un header simplu
df_g20 = pd.read_csv('g20.csv')

# Selectăm doar datele numerice (loadings)
# Matricea L ar trebui să fie de dimensiune (5 variabile x 2 factori)
L = df_g20.select_dtypes(include=[np.number]).values

# 1. Calculăm Comunalitatea (hi^2)
# Suma pătratelor elementelor de pe fiecare linie (pe cele 2 coloane de factori)
# axis=1 înseamnă suma pe orizontală
communalities = np.sum(L ** 2, axis=1)

# 2. Calculăm Varianța Specifică (epsilon)
# Pentru variabile standardizate, Varianța Totală = 1
# Varianța Specifică = 1 - Comunalitate
specific_variances = 1 - communalities

# 3. Identificăm indexul valorii maxime
# np.argmax returnează indexul (0, 1, 2...), deci adunăm 1 pentru a afișa "Variabila 1..5"
max_idx_zero_based = np.argmax(specific_variances)
variabila_max = max_idx_zero_based + 1
valoare_max = specific_variances[max_idx_zero_based]

print(f"Variabila cu cea mai mare varianță specifică este: {variabila_max}")
print(f"Valoarea varianței specifice: {valoare_max:.5f}")

# endregion

# region SUBIECT CALITATEA AERULUI

df_aer = pd.read_csv('CalitateaAeruluiTari.csv')

# 2. Definirea listei de indicatori (variabilele numerice)
indicatori = [
    'Air_quality_Carbon_Monoxide',
    'Air_quality_Ozone',
    'Air_quality_Nitrogen_dioxide',
    'Air_quality_Sulphur_dioxide',
    'Air_quality_PM2.5',
    'Air_quality_PM10'
]

# ==========================================
# CERINTA A1
# ==========================================

# 3. Calculul Coeficientului de Variație (CV)
# CV = Deviația Standard / Medie
# Calculăm media și deviația standard pentru fiecare coloană din lista de indicatori
means = df_aer[indicatori].mean()
stds = df_aer[indicatori].std()

# Calculăm raportul
cv_values = stds / means

# 4. Creare DataFrame pentru salvare
df_cv = pd.DataFrame({
    'Indicator': cv_values.index,
    'Coeficient_Variatie': cv_values.values
})

# 5. Salvare în fișierul Cerinta1.csv
output_path = './data_out/Cerinta1.csv'
df_cv.to_csv(output_path, index=False)
print("Cerinta A.1 salvată.")

# ==========================================
# CERINTA 2: Indicatorul cu variație maximă pe Continente
# ==========================================

# 1. Încărcare coduri țări
df_coduri = pd.read_csv('CoduriTari.csv')

# 2. Unire tabele (Merge) pe baza CountryId și Cod
# Inner join păstrează doar țările care au informații în ambele tabele
df_merged = pd.merge(df_aer, df_coduri[['Cod', 'Continent']],
                     left_on='CountryId', right_on='Cod', how='inner')

# 3. Calcul CV la nivel de Continent
# Grupăm după Continent și calculăm statisticile doar pentru indicatori
grouped = df_merged.groupby('Continent')[indicatori]

means_cont = grouped.mean()
stds_cont = grouped.std()

# Calculăm matricea coeficienților de variație (linii=Continente, coloane=Indicatori)
cv_cont = stds_cont / means_cont

# 4. Determinare maxim pe fiecare linie (Continent)
# idxmax(axis=1) returnează numele coloanei (indicatorului) cu valoarea maximă
max_indicators = cv_cont.idxmax(axis=1)
# max(axis=1) returnează valoarea maximă propriu-zisă
max_values = cv_cont.max(axis=1)

# 5. Construire DataFrame rezultat
df_result_c2 = pd.DataFrame({
    'Continent': cv_cont.index,
    'Indicator': max_indicators.values,
    'Valoare_CV': max_values.values
})

# 6. Salvare în fișierul Cerinta2.csv
df_result_c2.to_csv('./data_out/Cerinta2.csv', index=False)

# ==========================================
# SECTIUNEA B: Analiza în Componente Principale (ACP)
# ==========================================

# 1. Pregătirea datelor
X = df_aer[indicatori].values

# 2. Standardizarea datelor
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 3. Aplicarea PCA
pca = PCA()
pca.fit(X_std)

# --- B.1 Varianțele componentelor principale ---
explained_variance = pca.explained_variance_

for i, var in enumerate(explained_variance):
    print(f"Varianta componentei {i + 1}: {var:.5f}")

# --- B.2 Scorurile asociate instanțelor ---
# Calculul scorurilor (proiecțiile țărilor pe noile axe principale)
scores = pca.transform(X_std)

# Creare DataFrame pentru scoruri
# Generăm nume pentru componente: C1, C2, ..., C6
component_names = [f'C{i + 1}' for i in range(scores.shape[1])]
df_scores = pd.DataFrame(scores, columns=component_names)

# Adăugăm identificatorii pentru a ști cărei țări îi aparține scorul
# Presupunem că df_aer are coloanele 'CountryId' și 'Country' conform enunțului

df_scores['CountryId'] = df_aer['CountryId']
df_scores['Country'] = df_aer['Country']

# Reordonăm coloanele pentru a avea identificatorii la început
# Identificăm ce coloane de identificare avem disponibile
id_cols = [col for col in ['CountryId', 'Country'] if col in df_scores.columns]
cols_export = id_cols + component_names
df_scores = df_scores[cols_export]

# Salvare în fișierul scoruri.csv
output_scores = './data_out/scoruri.csv'
df_scores.to_csv(output_scores, index=False)

# --- B.3 Graficul scorurilor (C1 vs C2) ---
plt.figure(figsize=(10, 7))
plt.title('Graficul scorurilor în primele două axe principale')

# Adăugăm procentul de varianță explicată pe axe
var_ratio = pca.explained_variance_ratio_
plt.xlabel(f'Componenta 1 ({var_ratio[0] * 100:.2f}%)')
plt.ylabel(f'Componenta 2 ({var_ratio[1] * 100:.2f}%)')

# Plotare puncte
plt.scatter(scores[:, 0], scores[:, 1], c='steelblue', alpha=0.7, edgecolors='k')

# Adăugare etichete (Codul țării)
if 'CountryId' in df_aer.columns:
    for i, txt in enumerate(df_aer['CountryId']):
        # Adăugăm text doar pentru punctele vizibile sau importante pentru a evita aglomerarea
        # Aici punem pentru toate
        plt.text(scores[i, 0], scores[i, 1], str(txt), fontsize=8, ha='right', va='bottom')

# Linii axe prin origine
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()  # Decomentează pentru a afișa fereastra grafică

# ==========================================
# SECTIUNEA C: Analiza Discriminantă (Calcul pe valori proprii date)
# ==========================================

# Valori proprii date în enunț
eigenvalues = [0.9, 0.05, 0.8, 0.75, 0.5, 0.3, 0.2]

# 1. Sortare descrescătoare (important pentru cummulare)
eigenvalues_sorted = sorted(eigenvalues, reverse=True)

# 2. Suma totală
total_power = sum(eigenvalues_sorted)

# 3. Calcul procentaj cumulat
current_sum = 0
num_vars_needed = 0

print(f"Valori proprii sortate: {eigenvalues_sorted}")
print(f"Putere totală discriminare: {total_power}")

for val in eigenvalues_sorted:
    current_sum += val
    num_vars_needed += 1
    cum_percent = (current_sum / total_power) * 100

    # print(f"Var {num_vars_needed}: {val} -> {cum_percent:.2f}%")

    if cum_percent > 90:
        break

# endregion

# region SUBIECT E_NSAL_ANGAJATI

# Creare director output dacă nu există
if not os.path.exists('./data_out'):
    os.makedirs('./data_out')

df_employees = pd.read_csv('E_NSAL_2008-2021.csv')
df_population = pd.read_csv('PopulatieLocalitati.csv')

# ==========================================
# CERINTA 1: Anul cu maxim de angajați
# ==========================================
ani = [str(y) for y in range(2008, 2022)]

# Calcul
df_employees['An_Max_Angajati'] = df_employees[ani].idxmax(axis=1)

rezultat1 = df_employees[['SIRUTA', 'An_Max_Angajati']]
rezultat1.to_csv('./data_out/Cerinta1.csv', index=False)

# ==========================================
# CERINTA 2: Rata ocupării pe județe
# ==========================================

# 1. Unire tabele
df_merged = pd.merge(df_employees, df_population[['Siruta', 'Judet', 'Populatie']],
                     left_on='SIRUTA', right_on='Siruta', how='inner')

# 2. Agregare date la nivel de județ
agg_data = df_merged.groupby('Judet')[ani + ['Populatie']].sum()

# 3. Calcul Rata de Ocupare
rates_df = pd.DataFrame(index=agg_data.index)
for an in ani:
    rates_df[an] = agg_data[an] / agg_data['Populatie']

# 4. Calcul Rata Medie
rates_df['Rata_Medie'] = rates_df[ani].mean(axis=1)

# 5. Sortare descrescătoare
rates_df_sorted = rates_df.sort_values(by='Rata_Medie', ascending=False)

# 6. Salvare
rates_df_sorted.reset_index().to_csv('./data_out/Cerinta2.csv', index=False)

# ==========================================
# SECTIUNEA B: Analiza Liniară Discriminantă (LDA)
# ==========================================
# 1. Încărcare set antrenare
df_train = pd.read_csv('Pacienti.csv')

# 2. Definire variabile
predictors = ['L_CORE', 'L_SURF', 'L_02', 'L_BP', 'SURF_ST', 'CORE_ST', 'BP_ST']
target = 'DECISION'

X_train = df_train[predictors]
y_train = df_train[target]

# 3. Antrenare model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 4. Calcul scoruri
z_scores = lda.transform(X_train)

# 5. Salvare rezultate
cols_z = [f'Z{i + 1}' for i in range(z_scores.shape[1])]
df_z = pd.DataFrame(z_scores, columns=cols_z)
df_z['Id'] = df_train['Id']

# Reordonare (Id primul)
df_z = df_z[['Id'] + cols_z]

df_z.to_csv('./data_out/z.csv', index=False)

# B.2 Graficul scorurilor discriminante
plt.figure(figsize=(10, 7))
plt.title('Graficul scorurilor discriminante (Z1 vs Z2)')
plt.xlabel('Axa Discriminantă 1')
plt.ylabel('Axa Discriminantă 2')

# Creăm un dataframe temporar pentru plotare
df_plot = df_z.copy()
df_plot['Decizie'] = y_train.values  # Adăugăm etichetele claselor

# Scatter plot colorat după clasă
sns.scatterplot(data=df_plot, x='Z1', y='Z2', hue='Decizie', style='Decizie', s=100, palette='viridis')

# Adăugăm axele prin origine
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()

# B.3 Evaluare performanță
print("\n--- B.3 Evaluare performanță ---")

# Predicție
y_pred = lda.predict(X_train)

# Matricea de confuzie
cm = confusion_matrix(y_train, y_pred)
classes = lda.classes_
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
df_cm.to_csv('./data_out/matc.csv')
print(df_cm)

# Indicatori
accuracy = accuracy_score(y_train, y_pred)
report = classification_report(y_train, y_pred)  # zero_division=0 pentru a evita warning-uri la clase fără predicții

# endregion

# region SUBIECT ALCOOL

# 1. Încărcare Date

df_alcohol = pd.read_csv('alcohol.csv')
df_coduri = pd.read_csv('CoduriTariExtins.csv')  # Fișierul extins încărcat de tine
df_a = pd.read_csv('a.csv')  # Matricea vectorilor proprii

# Definire coloane ani
ani = ['2000', '2005', '2010', '2015', '2018']

# Tratare valori lipsă (înlocuire cu media pe coloană)
df_alcohol[ani] = df_alcohol[ani].fillna(df_alcohol[ani].mean())

# ==========================================
# CERINTA A
# ==========================================

# --- A.1 Media consumului pe cei 5 ani ---
df_alcohol['Consum_Mediu'] = df_alcohol[ani].mean(axis=1)
df_result_a1 = df_alcohol.sort_values(by='Consum_Mediu', ascending=False)

# Salvare Cerinta1.csv
output_a1 = df_result_a1[['Code', 'Country', 'Consum_Mediu']]
output_a1.to_csv('./data_out/Cerinta1.csv', index=False)

# --- A.2 Anul cu consum maxim pe continente ---

# Unire tabele. Verificăm dacă CoduriTariExtins are 'Code' sau doar 'Country'.
# Din snippet, pare să aibă 'Country' și 'Continent'. Facem merge pe 'Country'.
df_merged = pd.merge(df_alcohol, df_coduri, on='Country', how='inner')

# Calcul medii pe continent per an
continent_means = df_merged.groupby('Continent')[ani].mean()

# Identificare an maxim
max_years = continent_means.idxmax(axis=1)

# Salvare Cerinta2.csv
df_result_a2 = pd.DataFrame({
    'Continent Name': max_years.index,
    'Anul': max_years.values
})
df_result_a2.to_csv('./data_out/Cerinta2.csv', index=False)

# ==========================================
# CERINTA B: Clusteri (Ward)
# ==========================================

# Pregătire date (standardizare)
X = df_alcohol[ani].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Ierarhie (Ward)
Z = linkage(X_std, method='ward')

# --- B.1 Matricea Ierarhie ---
# Z conține: idx1, idx2, distanța, nr_instanțe
df_linkage = pd.DataFrame(Z, columns=['Cluster 1', 'Cluster 2', 'Distanta', 'Nr. Instante'])
print(df_linkage)

# --- B.2 Partiția în 5 clusteri ---
k = 5
clusters = hic.fcluster(Z, t=k, criterion='maxclust')
df_alcohol['Cluster'] = clusters

# Salvare p4.csv
output_p4 = df_alcohol[['Code', 'Country', 'Cluster']]
output_p4.to_csv('./data_out/p4.csv', index=False)

# --- B.3 Plot Clusteri ---
# Reducere la 2D cu PCA pentru vizualizare
pca = PCA(n_components=2)
components = pca.fit_transform(X_std)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap='viridis', alpha=0.8)
plt.title('Plotul partiției celor 5 clusteri (pe primele 2 axe principale)')
plt.xlabel('Componenta Principală 1')
plt.ylabel('Componenta Principală 2')
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True)
plt.show()

# ==========================================
# CERINTA C: Scoruri PCA pentru instanță
# ==========================================
# Instanța dată
x = np.array([3, 1, 2, 1, 4])

# Matricea A (vectori proprii)
A = df_a.values

# Calcul scoruri (proiecția pe axe)
# C = x * A
scores = np.dot(x, A)

print(f"Instanța x: {x}")
print("Scorurile obținute:")
print(scores)

# endregion

# region SUBIECT CU EMISIILE SI ENERGIA

# Încărcare date
df_emis = pd.read_csv('Emissions.csv')
df_elec = pd.read_csv('ElectricityProduction.csv')
df_pop = pd.read_csv('Population.csv')
df_c = pd.read_csv('c.csv')

# ==========================================
# CERINTA A
# ==========================================

# --- A.1: Emisii totale de particule ---

# Însumăm coloanele numerice de emisii
cols_emis = ['AirEmiss', 'Sulphur', 'Nitrogen', 'Ammonia', 'NonMeth', 'Partic', 'GreenGE', 'GreenGIE']
df_emis['Emisii_total_tone'] = df_emis[cols_emis].sum(axis=1)

output_a1 = df_emis[['CountryCode', 'Country', 'Emisii_total_tone']]
output_a1.to_csv('./data_out/Cerinta1.csv', index=False)

# --- A.2: Emisii la 100.000 locuitori pe Regiuni ---
# Unire cu populația
df_merged_a2 = pd.merge(df_emis, df_pop, on='CountryCode', how='inner')

# Agregare pe regiune
region_groups = df_merged_a2.groupby('Region')[cols_emis + ['Population']].sum()

# Calcul per 100.000 locuitori
df_result_a2 = pd.DataFrame(index=region_groups.index)
for col in cols_emis:
    df_result_a2[col] = (region_groups[col] / region_groups['Population']) * 100000

df_result_a2.reset_index(inplace=True)
df_result_a2.to_csv('./data_out/Cerinta2.csv', index=False)

# ==========================================
# CERINTA B: Analiza Canonică (CCA)
# ==========================================

# 1. Unire tabele
df_cca = pd.merge(df_elec, df_emis, on='CountryCode', how='inner')

# Variabile
vars_x = ['coal', 'oil', 'gas', 'nuclear', 'hydro', 'biomass', 'waste', 'other']
vars_y = ['AirEmiss', 'Sulphur', 'Nitrogen', 'Ammonia', 'NonMeth', 'Partic', 'GreenGE', 'GreenGIE']

X = df_cca[vars_x].values
Y = df_cca[vars_y].values

# Standardizare
scaler_x = StandardScaler()
X_std = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
Y_std = scaler_y.fit_transform(Y)

# Aplicare CCA
n_comps = min(X.shape[1], Y.shape[1], X.shape[0])
cca = CCA(n_components=n_comps)
cca.fit(X_std, Y_std)

# --- B.1 Calcul Scoruri ---
X_c, Y_c = cca.transform(X_std, Y_std)

df_z = pd.DataFrame(X_c, columns=[f'CanX_{i + 1}' for i in range(n_comps)])
df_z.to_csv('./data_out/z.csv', index=False)

df_u = pd.DataFrame(Y_c, columns=[f'CanY_{i + 1}' for i in range(n_comps)])
df_u.to_csv('./data_out/u.csv', index=False)

# --- B.2 Corelații Canonice ---
canonical_corrs = []
for i in range(n_comps):
    # Calculăm corelația Pearson între perechile de variabile canonice
    corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
    canonical_corrs.append(corr)

df_r = pd.DataFrame(canonical_corrs, columns=['Correlation'])
df_r.to_csv('./data_out/r.csv', index=False)

# --- B.3 Testul Bartlett ---
n_obs = X.shape[0]
p = X.shape[1]
q = Y.shape[1]

print(f"{'Radacina':<10} {'Corelatie':<12} {'Wilks L':<10} {'Chi-sq':<10} {'p-value':<10}")

significant_roots = 0
for k in range(n_comps):
    # Calcul Lambda Wilks folosind corelațiile rămase
    rs_sq = [r ** 2 for r in canonical_corrs[k:]]
    wilks = np.prod([1 - r2 for r2 in rs_sq])

    # Chi-square statistic
    chi_val = -(n_obs - 1 - (p + q + 1) / 2) * np.log(wilks)

    # Grade de libertate
    df = (p - k) * (q - k)

    # P-value
    p_val = 1 - chi2.cdf(chi_val, df)

    if p_val < 0.01:
        significant_roots += 1

    print(f"{k + 1:<10} {canonical_corrs[k]:<12.4f} {wilks:<10.4f} {chi_val:<10.4f} {p_val:<10.4e}")

print(f"\nNumăr rădăcini semnificative (p < 0.01): {significant_roots}")

# ==========================================
# CERINTA C: Criteriul Cattell (PCA)
# ==========================================

# Calculăm valorile proprii din scoruri
scores_c = df_c.values
eigenvalues_c = np.var(scores_c, axis=0)

# Sortare
eigenvalues_c = sorted(eigenvalues_c, reverse=True)

print("Valorile proprii (Varianțe):")
for i, ev in enumerate(eigenvalues_c):
    print(f"Componenta {i + 1}: {ev:.4f}")

# Pentru Cattell, se caută vizual punctul de inflexiune ("cotul").
# Aici afișăm doar valorile, lăsând interpretarea utilizatorului,
# sau numărăm câte sunt supraunitare (Kaiser) ca referință.
cnt = sum(i > 1 for i in eigenvalues_c)
print(f"Număr componente semnificative (Ref. Kaiser > 1): {cnt}")

# endregion

# # region SUBIECT INDICATORI GLOBALI PER CAPITA
#
# # Încărcare date
# df_ind = pd.read_csv('GlobalIndicatorsPerCapita_2021.csv')
# df_cod = pd.read_csv('CoduriTari.csv')
#
# # =============================================================================
# # CERINTA A
# # =============================================================================
#
# # --- A.1: Excedent Comercial ---
# # Calcul diferență
# df_ind['Excedent'] = df_ind['Exports'] - df_ind['Imports']
#
# # Filtrare excedent pozitiv
# df_excedent = df_ind[df_ind['Excedent'] > 0].copy()
#
# # Sortare descrescătoare
# df_excedent = df_excedent.sort_values(by='Excedent', ascending=False)
#
# # Selectare coloane și salvare
# output_cols = ['CountryID', 'Country', 'Exports', 'Imports', 'Excedent']
# df_excedent[output_cols].to_csv('./data_out/Cerinta1.csv', index=False)
#
# # --- A.2: Matrice de corelație pe Continente ---
# # Unire tabele pentru a obține continentul
# df_merged = pd.merge(df_ind, df_cod[['CountryID', 'Continent']], on='CountryID', how='inner')
#
# # Identificare coloane numerice (indicatorii)
# # Excludem ID, Country, Excedent, Continent
# numeric_cols = indicators  # Lista definită la generare
#
# # Grupare după continent
# groups = df_merged.groupby('Continent')
#
# for continent_name, group_data in groups:
#     # Calcul matrice corelație pentru indicatori
#     corr_matrix = group_data[numeric_cols].corr()
#
#     # Salvare fișier
#     file_name = f"./data_out/{continent_name}.csv"
#     corr_matrix.to_csv(file_name)
#
# # =============================================================================
# # CERINTA B: Analiza în Componente Principale (PCA)
# # =============================================================================
#
# # 1. Pregătirea datelor
# X = df_ind[indicators].values
# labels = df_ind['Country'].values
#
# # 2. Standardizare
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
#
# # 3. Aplicare PCA
# pca = PCA()
# pca.fit(X_std)
#
# # --- B.1: Scree Plot (Graficul varianței) ---
# eigenvalues = pca.explained_variance_
# components = np.arange(1, len(eigenvalues) + 1)
#
# plt.figure(figsize=(10, 6))
# plt.plot(components, eigenvalues, marker='o', linestyle='-', color='blue')
# plt.axhline(y=1, color='r', linestyle='--', label='Criteriul Kaiser (Valoare=1)')
# plt.title('Scree Plot - Varianța Componentelor Principale')
# plt.xlabel('Componenta Principală')
# plt.ylabel('Valoare Proprie (Eigenvalue)')
# plt.legend()
# plt.grid(True)
# plt.show()  # Decomentează dacă rulezi local cu interfață grafică
#
# # --- B.2: Corelațiile Factoriale (Factor Loadings) ---
# # Formula: r(X, C) = eigenvector * sqrt(eigenvalue)
# # pca.components_ conține vectorii proprii pe linii (transpusă față de notația matematică standard X * A)
# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
#
# # Creare DataFrame pentru lizibilitate
# col_names_pca = [f'PC{i + 1}' for i in range(len(eigenvalues))]
# df_loadings = pd.DataFrame(loadings, index=indicators, columns=col_names_pca)
#
# df_loadings.to_csv('./data_out/r.csv')
#
# # --- B.3: Cosinus și Comunalități ---
# # Cosinus pătrat (Calitatea reprezentării) = Loading^2
# # Comunalitatea = Suma (Loading^2) pe componentele reținute.
# # Aici calculăm pentru toate componentele generate.
#
# df_cosin = df_loadings ** 2
# df_cosin.to_csv('./data_out/cosin.csv')
#
# # Comunalități (suma pe linii a cosinusurilor pătrate)
# # Într-o analiză PCA completă, comunalitatea este 1. Dacă se rețin k componente, e suma primelor k.
# # Salvăm comunalitățile totale calculate pe toate componentele (ar trebui să fie aprox 1).
# communalities = df_cosin.sum(axis=1)
# df_comm = pd.DataFrame(communalities, columns=['Communality'])
# df_comm.to_csv('./data_out/comm.csv')
#
# # =============================================================================
# # CERINTA C: Analiza Discriminantă (Clasificare)
# # =============================================================================
# # Încărcare date
# # T.csv nu are header
# T_matrix = pd.read_csv('T.csv', header=None).values
# # G.csv are header
# df_g = pd.read_csv('G.csv')
#
# # Extragere predictori și clase
# # Prima coloană e clasa, restul X1...X6
# centers = df_g.iloc[:, 1:].values
# class_labels = df_g.iloc[:, 0].values
#
# # Instanța de clasificat
# x = np.array([118, 19.8, 2.5, 8.3, 61.1, 8.3])
#
# # Calcul inversa matricei de covarianță
# inv_T = np.linalg.inv(T_matrix)
#
# # Calcul distanță Mahalanobis (sau scor discriminant)
# # Distanța(x, g) = (x - g)^T * T^-1 * (x - g)
# # Atribuim clasa cu distanța minimă
#
# min_dist = float('inf')
# predicted_class = None
#
# print("Distanțele calculate:")
# for i in range(len(class_labels)):
#     g = centers[i]
#     diff = x - g
#     # Calcul distanță
#     dist = np.dot(np.dot(diff.T, inv_T), diff)
#
#     print(f"Clasa {class_labels[i]}: Distanța = {dist:.4f}")
#
#     if dist < min_dist:
#         min_dist = dist
#         predicted_class = class_labels[i]
#
# print(f"\nInstanța x a fost clasificată în clasa: {predicted_class}")
#
# # endregion
#
# # region SUBIECT RATA SPORULUI NATURAL
#
# # Încărcare date
# df_pop = pd.read_csv('Population.csv')
# df_cont = pd.read_csv('Continents.csv')
#
# # =============================================================================
# # CERINTA A
# # =============================================================================
# # --- A.1: Țări cu RS < Media Globală ---
#
# # Calculul mediei globale pentru RS
# mean_rs = df_pop['RS'].mean()
#
# # Filtrare țări
# df_a1 = df_pop[df_pop['RS'] < mean_rs].copy()
#
# # Sortare descrescătoare după RS
# df_a1 = df_a1.sort_values(by='RS', ascending=False)
#
# # Selectare coloane
# output_a1 = df_a1[['Three_Letter_Country_Code', 'Country_Name', 'RS']]
#
# # Salvare
# output_a1.to_csv('./data_out/Cerinta1.csv', index=False)
#
# # --- A.2: Maxime pe Continente ---
#
# # Unire tabele pentru a avea continentul
# df_merged = pd.merge(df_pop, df_cont, on='Three_Letter_Country_Code', how='inner')
#
# # Lista de indicatori pentru care căutăm maximul
# indicators_list = ['RS', 'FR', 'IM', 'LE', 'LEF', 'LEM', 'MMR']
#
# # Grupăm după continent
# grouped = df_merged.groupby('Continent_Name')
#
# # Pregătim lista pentru rezultate
# results_a2 = []
#
# for continent, group in grouped:
#     row = {'Continent_Name': continent}
#     # Pentru fiecare indicator, găsim codul țării cu valoarea maximă
#     for ind in indicators_list:
#         # idxmax returnează indexul liniei cu valoarea maximă
#         idx_max = group[ind].idxmax()
#         country_code = group.loc[idx_max, 'Three_Letter_Country_Code']
#         row[ind] = country_code
#     results_a2.append(row)
#
# df_a2 = pd.DataFrame(results_a2)
#
# # Reordonăm coloanele conform exemplului (Continent, apoi indicatorii)
# cols_order = ['Continent_Name'] + indicators_list
# df_a2 = df_a2[cols_order]
#
# # Salvare
# df_a2.to_csv('./data_out/Cerinta2.csv', index=False)
#
# # =============================================================================
# # CERINTA B: Analiza în Componente Principale (PCA)
# # =============================================================================
#
# # 1. Pregătirea datelor (doar variabilele numerice)
# X = df_pop[indicators].values
# variable_names = indicators
#
# # 2. Standardizare
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
#
# # 3. Aplicare PCA
# pca = PCA()
# pca.fit(X_std)
#
# # --- B.1: Tabelul Varianței ---
# variance = pca.explained_variance_  # Valori proprii (lambda)
# variance_ratio = pca.explained_variance_ratio_  # Procent varianta explicata
# cum_variance = np.cumsum(variance)  # Varianta cumulata
# cum_ratio = np.cumsum(variance_ratio)  # Procent cumulat
#
# # Creare DataFrame
# df_varianta = pd.DataFrame({
#     'Varianta': variance,
#     'Varianta Cumulata': cum_variance,
#     'Procent Varianta': variance_ratio * 100,
#     'Procent Cumulat': cum_ratio * 100
# })
#
# # Salvare
# df_varianta.to_csv('./data_out/Varianta.csv', index=False)
#
# # --- B.2: Plotul de varianță (Scree Plot) ---
# components = np.arange(1, len(variance) + 1)
#
# plt.figure(figsize=(10, 6))
# plt.plot(components, variance, 'bo-', linewidth=2, markersize=8)
# plt.title('Scree Plot - Varianța Componentelor Principale')
# plt.xlabel('Componenta')
# plt.ylabel('Valoare Proprie (Eigenvalue)')
# plt.xticks(components)
#
# # Marcarea criteriului Kaiser (Valoare proprie >= 1)
# plt.axhline(y=1, color='r', linestyle='--', label='Criteriul Kaiser (y=1)')
# plt.legend()
# plt.grid(True)
# plt.show() # Decomenteaza daca vrei sa vezi graficul
#
# # --- B.3: Corelograma (Heatmap Corelații Variabile - Componente) ---
# # Calculăm corelațiile factoriale (Factor Loadings)
# # Formula: Loading = Eigenvector * sqrt(Eigenvalue)
# # pca.components_ sunt vectorii proprii (pe linii), deci transpunem
# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
#
# # Etichete componente
# comp_labels = [f'PC{i + 1}' for i in range(len(variance))]
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(loadings, annot=True, cmap='RdBu', center=0,
#             xticklabels=comp_labels, yticklabels=variable_names, vmin=-1, vmax=1)
# plt.title('Corelograma: Corelații Variabile - Componente Principale')
# plt.show() # Decomenteaza daca vrei sa vezi graficul
#
# # =============================================================================
# # CERINTA C: Analiza Discriminantă (LDA)
# # =============================================================================
#
# # Valori proprii date
# eigenvalues_lda = [0.9, 0.05, 0.8, 0.75, 0.5, 0.3, 0.2]
#
# # 1. Sortare descrescătoare (cea mai mare putere de discriminare prima)
# eigenvalues_sorted = sorted(eigenvalues_lda, reverse=True)
#
# # 2. Calcul suma totală
# total_power = sum(eigenvalues_sorted)
#
# # 3. Determinare număr minim pentru > 90%
# current_sum = 0
# min_vars = 0
#
# print(f"Valori proprii sortate: {eigenvalues_sorted}")
# print(f"Putere totală: {total_power}")
#
# for val in eigenvalues_sorted:
#     current_sum += val
#     min_vars += 1
#
#     # Calcul procent cumulat
#     cum_percent = (current_sum / total_power) * 100
#
#     # Verificare prag
#     if cum_percent > 90:
#         print(f"Prag atins la variabila {min_vars} cu un procent cumulat de {cum_percent:.2f}%")
#         break
#
# print(f"Numărul minim de variabile discriminante necesare: {min_vars}")
#
# # endregion