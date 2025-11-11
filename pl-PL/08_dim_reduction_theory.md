Jasne, oto szczegółowe omówienie rozdziału 8 "Redukcja Wymiarowości", przygotowane zgodnie z Twoimi wytycznymi. Tekst został przetworzony tak, aby skupić się na kluczowej wiedzy, wyjaśnieniach "dlaczego coś robimy" oraz praktycznych zastosowaniach w prosty i zrozumiały sposób.

### **Rozdział 8: Redukcja Wymiarowości**

#### **Wprowadzenie do problemu**

**Teoria i Przeznaczenie:**
Wiele problemów w uczeniu maszynowym operuje na danych z ogromną liczbą cech (wymiarów). Wyobraź sobie zdjęcie 28x28 pikseli z zestawu MNIST – każda instancja (zdjęcie) ma 784 cechy (każdy piksel to jedna cecha). Taka ilość danych powoduje dwa główne problemy:
1.  **Wydajność:** Trenowanie modelu na milionach cech jest niezwykle powolne i wymaga ogromnych zasobów obliczeniowych.
2.  **Jakość rozwiązania:** Przy zbyt wielu cechach modelowi trudniej jest znaleźć ogólne wzorce. Może on zacząć "uczyć się na pamięć" szumu i nieistotnych detali, zamiast generalizować wiedzę. Zjawisko to nazywane jest **klątwą wymiarowości**.

**Cel redukcji wymiarowości:** Celem jest zmniejszenie liczby cech w zbiorze danych, przy jednoczesnym zachowaniu jak największej ilości istotnych informacji. To trochę jak robienie streszczenia długiej książki – wyrzucamy mniej ważne wątki, aby skupić się na głównej fabule.

**Korzyści:**
*   **Przyspieszenie treningu:** Mniej danych do przetworzenia oznacza szybsze działanie algorytmów.
*   **Wizualizacja danych (DataViz):** Człowiek nie jest w stanie wyobrazić sobie przestrzeni o więcej niż 3 wymiarach. Redukując dane do 2 lub 3 wymiarów, możemy je narysować na wykresie i wizualnie zidentyfikować klastry, trendy i inne wzorce, co jest bezcenne przy analizie i prezentacji wyników.
*   **Potencjalna poprawa modelu:** Czasami usunięcie szumu i nieistotnych cech może sprawić, że model będzie działał lepiej, bo skupi się na tym, co naprawdę ważne.

**Wady:**
*   **Utrata informacji:** Każda redukcja to kompromis. Zawsze tracimy część informacji, co może nieznacznie pogorszyć działanie modelu.
*   **Zwiększona złożoność:** Dodajemy kolejny krok do naszego procesu przetwarzania danych, co może utrudnić utrzymanie projektu.

---

### **Podrozdział 1: Klątwa Wymiarowości (The Curse of Dimensionality)**

**Teoria i Przeznaczenie:**
To zbiór problemów, które pojawiają się podczas pracy z danymi w przestrzeniach o wysokiej wymiarowości. Nasza intuicja, ukształtowana w świecie 3D, zawodzi.

**Kluczowe problemy:**
1.  **Dane stają się "puste" (sparse):** Wraz ze wzrostem liczby wymiarów, punkty danych oddalają się od siebie. Wyobraź sobie kwadrat 1x1. Średni dystans między dwoma losowymi punktami jest mały. Teraz wyobraź sobie hipersześcian o 10 000 wymiarach. Średni dystans między dwoma punktami staje się ogromny, mimo że wszystkie mieszczą się w tej samej "jednostkowej" przestrzeni.
    *   **Dlaczego to problem?** Jeśli nowy punkt, dla którego chcemy zrobić predykcję, jest bardzo daleko od wszystkich punktów treningowych, model musi dokonać ogromnej "ekstrapolacji" (zgadywania w nieznanym obszarze). Taka predykcja jest bardzo niepewna i podatna na błędy. Model ma wysokie ryzyko **nadmiernego dopasowania (overfittingu)**, ponieważ trudno mu znaleźć ogólne trendy w tak rozproszonych danych.

2.  **Większość punktów leży blisko granicy:** W 2D (kwadrat) jest mało prawdopodobne, że losowy punkt znajdzie się bardzo blisko krawędzi. W 10 000 wymiarów jest to niemal pewne. Oznacza to, że dane treningowe słabo reprezentują środek przestrzeni.

3.  **Wymagana wykładnicza ilość danych:** Aby zachować tę samą "gęstość" punktów danych (czyli żeby punkty nie były od siebie za daleko), wraz z każdym nowym wymiarem potrzebujemy wykładniczo więcej danych. W praktyce jest to niemożliwe do osiągnięcia.

**Cel zrozumienia klątwy wymiarowości:** Uświadomienie sobie, że dodawanie kolejnych cech nie zawsze jest dobre. Czasem mniej znaczy więcej, a redukcja wymiarowości jest koniecznością, a nie tylko optymalizacją.

---

### **Podrozdział 2: Główne Podejścia do Redukcji Wymiarowości**

Istnieją dwie główne strategie walki z klątwą wymiarowości.

#### **1. Projekcja (Projection)**

**Teoria i Przeznaczenie:**
Projekcja działa na założeniu, że chociaż dane mają wiele wymiarów, w rzeczywistości układają się wzdłuż znacznie niżej wymiarowej "płaszczyzny" lub podprzestrzeni.

*   **Jak to działa?** Wyobraź sobie, że rzucasz latarką na obiekty 3D, a na ścianie powstaje ich dwuwymiarowy cień. Projekcja to matematyczny odpowiednik tworzenia takiego "cienia". Znajdujemy odpowiednią "ścianę" (podprzestrzeń) i "rzutujemy" na nią wszystkie punkty danych.
*   **Kiedy używać?** Gdy dane mają strukturę zbliżoną do liniowej (np. punkty w 3D tworzące chmurę w kształcie płaskiego dysku). Wtedy rzutowanie na płaszczyznę tego dysku zachowa większość informacji.
*   **Przykład:** Dane o domach mogą mieć 100 cech, ale może się okazać, że większość zmienności (informacji) wynika z kombinacji kilku z nich, jak `powierzchnia` i `lokalizacja`. Możemy sprowadzić te 100 cech do 2-3 nowych, "syntetycznych" cech, które reprezentują te główne trendy.

#### **2. Uczenie Rozmaitości (Manifold Learning)**

**Teoria i Przeznaczenie:**
To podejście jest stosowane, gdy dane leżą na zakrzywionej, nieregularnej powierzchni w wyższej wymiarowości, zwanej **rozmaitością (manifold)**.

*   **Czym jest rozmaitość?** Wyobraź sobie zwinięty w rolkę kawałek papieru (w przykładzie z książki tzw. "Swiss roll"). Globalnie jest to obiekt 3D. Ale jeśli spojrzysz na dowolny mały fragment tej rolki, wygląda on jak płaska powierzchnia 2D. Ziemia jest kolejnym przykładem – globalnie jest kulą (3D), ale lokalnie postrzegamy ją jako płaską (2D). Rozmaitość to właśnie taka przestrzeń, która lokalnie przypomina płaską przestrzeń o niższej wymiarowości.
*   **Założenie o rozmaitości (Manifold Assumption):** Wiele rzeczywistych zbiorów danych (np. obrazy twarzy, cyfr) nie jest losowo rozrzuconych w przestrzeni, ale skupia się na takiej właśnie nisko wymiarowej, zakrzywionej rozmaitości.
*   **Jak to działa?** Algorytmy Manifold Learning nie próbują "zgnieść" danych przez projekcję. Zamiast tego próbują je "rozwinąć" lub "rozprostować". Dla "Swiss roll" celem jest uzyskanie płaskiego prostokąta 2D.
*   **Kiedy używać?** Gdy podejrzewamy, że nasze dane mają złożoną, nieliniową strukturę.

---

### **Podrozdział 3: PCA (Principal Component Analysis - Analiza Głównych Składowych)**

PCA to najpopularniejsza technika redukcji wymiarowości, oparta na **projekcji**.

**Teoria i Przeznaczenie:**
Celem PCA jest znalezienie takiej płaszczyzny (lub ogólniej: hiperpłaszczyzny), na którą rzutowanie danych powoduje jak najmniejszą utratę informacji.

*   **Jak PCA mierzy "informację"?** Informacja jest utożsamiana z **wariancją** (rozrzutem) danych. Oś, wzdłuż której dane są najbardziej rozrzucone, przechowuje najwięcej informacji. Oś, wzdłuż której punkty są mocno ściśnięte, jest mniej ważna.
*   **Jak to działa?**
    1.  PCA znajduje oś w danych, która ma największą wariancję. Nazywa ją **pierwszą główną składową (1st Principal Component, PC1)**.
    2.  Następnie szuka drugiej osi, która jest prostopadła (ortogonalna) do pierwszej i ma największą z pozostałej wariancji. To jest **PC2**.
    3.  Proces jest powtarzany, aż znajdziemy tyle osi, ile wymiarów miały oryginalne dane.
*   **Redukcja:** Aby zredukować wymiarowość do *d* wymiarów, po prostu wybieramy *d* pierwszych głównych składowych (tych z największą wariancją) i rzutujemy na nie dane. Te *d* osi tworzy nową podprzestrzeń.

#### **Wybór odpowiedniej liczby wymiarów**

**Teoria i Przeznaczenie:**
Skąd wiedzieć, czy zredukować dane do 2, 5 czy 50 wymiarów? Nie wybieramy tej liczby na ślepo. Chcemy zachować określoną ilość informacji (wariancji).

*   **Współczynnik wyjaśnionej wariancji (Explained Variance Ratio):** To metryka, która mówi, jaki procent całkowitej wariancji danych jest "przechowywany" przez każdą główną składową. Na przykład, jeśli dla PC1 wynosi on 84%, a dla PC2 14%, to te dwie osie razem wyjaśniają 98% zmienności w danych. Oznacza to, że redukcja do 2D jest w tym przypadku bardzo dobra.
*   **Jak wybrać?** Zamiast podawać konkretną liczbę wymiarów, możemy powiedzieć PCA: "zredukuj dane tak, aby zachować 95% oryginalnej wariancji". Algorytm sam dobierze minimalną liczbę głównych składowych potrzebną do osiągnięcia tego celu.

**Implementacja w Scikit-Learn:**
`PCA` to klasa, która robi to wszystko za nas.

```python
from sklearn.decomposition import PCA

# Inicjalizujemy PCA, mówiąc mu, by zachował 95% wariancji
# Zamiast liczby całkowitej (np. n_components=2), podajemy ułamek.
pca = PCA(n_components=0.95)

# pca.fit_transform(X) wykonuje dwa kroki:
# 1. fit(X): Analizuje dane X, uczy się, gdzie są główne składowe (osie z największą wariancją).
# 2. transform(X): Rzutuje oryginalne dane X na te nowo znalezione osie.
X_reduced = pca.fit_transform(X_train)

# Po wykonaniu fit, możemy sprawdzić, ile wymiarów zostało wybranych:
print(pca.n_components_) 
# oraz ile wariancji wyjaśnia każda z nich:
print(pca.explained_variance_ratio_)
```
*   **Dlaczego `fit_transform`?** Proces uczenia się głównych składowych (`fit`) i transformacji danych (`transform`) są często wykonywane razem na zbiorze treningowym. `fit_transform` jest zoptymalizowaną metodą, która robi obie te rzeczy jednocześnie. Na zbiorze testowym użyjemy już **tylko `transform`**, ponieważ osie zostały już zdefiniowane na podstawie danych treningowych – nie chcemy ich uczyć od nowa, bo to byłoby "oszukiwanie" (wyciek informacji ze zbioru testowego).

#### **PCA do kompresji danych**

**Teoria i Przeznaczenie:**
Redukcja wymiarowości to forma kompresji stratnej (jak w plikach JPEG). Zmniejszamy rozmiar danych, ale tracimy trochę jakości. PCA pozwala również na dekompresję.

*   **Jak działa dekompresja?** Możemy użyć metody `inverse_transform`, aby odtworzyć dane w oryginalnej, wysoko wymiarowej przestrzeni.
*   **Wynik:** Odtworzone dane nie będą identyczne z oryginałem, ponieważ informacja (wariancja) z odrzuconych osi została bezpowrotnie utracona. Różnica między oryginałem a odtworzonymi danymi to **błąd rekonstrukcji**.
```python
# Załóżmy, że pca został już wytrenowany (metodą fit)
X_recovered = pca.inverse_transform(X_reduced)
```
*   **Przeznaczenie:** Możemy użyć błędu rekonstrukcji do oceny, jak dobra była nasza redukcja. Niski błąd oznacza, że zachowaliśmy większość ważnych informacji.

#### **Warianty PCA dla dużych zbiorów danych**

Standardowe PCA wymaga załadowania całego zbioru danych do pamięci RAM, co jest niemożliwe dla bardzo dużych danych.

1.  **Incremental PCA (IPCA - Przyrostowe PCA):**
    *   **Przeznaczenie:** Do przetwarzania zbiorów danych, które nie mieszczą się w pamięci, lub do uczenia "online" (gdy dane napływają strumieniowo).
    *   **Jak działa?** Dzieli dane na małe porcje (mini-batche) i "karmi" nimi algorytm kawałek po kawałku, aktualizując główne składowe po każdej porcji.
    *   **Implementacja:** Zamiast `fit()`, używamy metody `partial_fit()` w pętli dla każdej porcji danych.

    ```python
    from sklearn.decomposition import IncrementalPCA

    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154) # np. dla MNIST

    # Dzielimy dane na 100 części
    for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch) # Uczymy się na małym kawałku

    X_reduced = inc_pca.transform(X_train) # Transformujemy całość na końcu
    ```

2.  **Randomized PCA (Zrandomizowane PCA):**
    *   **Przeznaczenie:** Znacznie szybsza alternatywa dla standardowego PCA, gdy docelowa liczba wymiarów *d* jest znacznie mniejsza niż oryginalna *n*.
    *   **Jak działa?** Używa losowego algorytmu stochastycznego do znalezienia *przybliżonych* głównych składowych. Zazwyczaj daje bardzo dobre wyniki w ułamku czasu.
    *   **Implementacja:** Wystarczy ustawić `svd_solver="randomized"` w konstruktorze PCA.

---

### **Podrozdział 4: Kernel PCA (kPCA - Jądrowe PCA)**

**Teoria i Przeznaczenie:**
Standardowe PCA jest algorytmem liniowym. Oznacza to, że nie radzi sobie z danymi o złożonych, nieliniowych strukturach (jak wspomniany "Swiss roll"). kPCA jest rozszerzeniem PCA, które rozwiązuje ten problem.

*   **Jak to działa?** kPCA wykorzystuje **sztuczkę jądrową (kernel trick)**, znaną z maszyn SVM.
    1.  Dane są w sposób niejawny (bez faktycznego obliczania) mapowane do przestrzeni cech o nieskończenie wielu wymiarach.
    2.  W tej nowej, wysoko wymiarowej przestrzeni, złożone nieliniowe zależności stają się liniowe.
    3.  W tej przestrzeni wykonywane jest standardowe, liniowe PCA.
    4.  Wynik jest rzutowany z powrotem do niższej wymiarowości.
*   **Efekt:** W oryginalnej przestrzeni danych uzyskujemy nieliniową redukcję wymiarowości, która potrafi "rozwinąć" skomplikowane rozmaitości.

#### **Wybór jądra i hiperparametrów**

**Teoria i Przeznaczenie:**
kPCA jest algorytmem nienadzorowanym, więc nie mamy prostych metryk (jak dokładność), aby wybrać najlepsze jądro (`kernel`) i jego hiperparametry (np. `gamma`). Mamy dwa główne podejścia:

1.  **Podejście zadaniowe (najpopularniejsze):**
    *   **Cel:** Redukcja wymiarowości jest często tylko krokiem przygotowawczym przed właściwym zadaniem, np. klasyfikacją. Możemy więc ocenić jakość kPCA na podstawie tego, jak dobrze działa finalny model.
    *   **Jak to zrobić?** Tworzymy `Pipeline`, czyli potok składający się z dwóch kroków: kPCA, a następnie klasyfikator (np. regresja logistyczna). Następnie używamy `GridSearchCV`, aby przeszukać różne jądra i wartości gamma dla kPCA. `GridSearchCV` wybierze takie parametry, które dadzą najwyższą dokładność klasyfikacji na końcu potoku.

    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    
    # Tworzymy potok: najpierw kPCA, potem klasyfikator
    clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

    # Definiujemy siatkę parametrów do przetestowania dla kroku "kpca"
    param_grid = [{
        "kpca__kernel": ["rbf", "sigmoid"],
        "kpca__gamma": np.linspace(0.03, 0.05, 10)
    }]

    # GridSearch znajdzie najlepszą kombinację
    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    ```
    *   **Dlaczego `kpca__gamma`?** Podwójny podkreślnik w nazwie parametru w `param_grid` mówi `GridSearchCV`, że parametr `gamma` dotyczy kroku o nazwie `kpca` w naszym `Pipeline`.

2.  **Podejście nienadzorowane (błąd rekonstrukcji):**
    *   **Cel:** Znaleźć parametry kPCA, które minimalizują błąd rekonstrukcji. Rekonstrukcja w kPCA jest trudniejsza niż w PCA, ponieważ musimy odtworzyć punkt z nieskończenie wymiarowej przestrzeni. Punkt ten nazywa się **pre-image**.
    *   **Jak to zrobić?** Scikit-Learn potrafi to zrobić, jeśli ustawimy `fit_inverse_transform=True` w `KernelPCA`. Możemy wtedy użyć błędu rekonstrukcji jako metryki do optymalizacji hiperparametrów.

---

### **Podrozdział 5: LLE (Locally Linear Embedding)**

**Teoria i Przeznaczenie:**
LLE to kolejna potężna, nieliniowa technika redukcji wymiarowości, należąca do rodziny **Manifold Learning**. W przeciwieństwie do PCA i kPCA, nie opiera się na projekcji.

*   **Jak to działa?** LLE działa w dwóch krokach:
    1.  **Znajdowanie lokalnych zależności:** Dla każdego punktu danych, LLE identyfikuje jego *k* najbliższych sąsiadów. Następnie próbuje odtworzyć ten punkt jako liniową kombinację tych sąsiadów. To pozwala zapisać "lokalną geometrię" danych.
    2.  **Zachowanie zależności w niskim wymiarze:** LLE tworzy nową, nisko wymiarową reprezentację danych, starając się, aby te same lokalne zależności (relacje z sąsiadami) zostały jak najlepiej zachowane.
*   **Przeznaczenie:** LLE jest szczególnie dobre w "rozwijaniu" pozwijanych rozmaitości, pod warunkiem, że nie ma w nich zbyt wiele szumu, a dane są w miarę gęsto rozmieszczone.

```python
from sklearn.manifold import LocallyLinearEmbedding

# n_neighbors to kluczowy hiperparametr - ilu sąsiadów brać pod uwagę
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
```

---

### **Podrozdział 6: Inne Techniki Redukcji Wymiarowości**

*   **MDS (Multidimensional Scaling):** Redukuje wymiarowość, starając się zachować odległości między instancjami.
*   **Isomap:** Rozwija rozmaitość, próbując zachować **odległość geodezyjną** (drogę "po powierzchni" rozmaitości, a nie na skróty przez pustą przestrzeń).
*   **t-SNE:** Najczęściej używane do **wizualizacji**. Działa tak, aby podobne instancje były na wykresie blisko siebie, a niepodobne daleko. Tworzy bardzo czytelne i estetyczne wizualizacje klastrów.
*   **LDA (Linear Discriminant Analysis):** To algorytm **nadzorowany**. W przeciwieństwie do PCA, które szuka osi maksymalnej wariancji, LDA szuka osi, które **najlepiej separują od siebie klasy**. Jest świetnym narzędziem do redukcji wymiarowości jako krok wstępny przed innym klasyfikatorem, ponieważ jawnie dba o to, by klasy pozostały rozdzielone po projekcji.

---

### **Schematyczny Plan Projektu z Redukcją Wymiarowości ("Przepis")**

Oto kroki, które należy wykonać, realizując projekt z redukcją wymiarowości:

**Krok 1: Zdefiniuj Cel**
*   Czy celem jest przyspieszenie treningu modelu?
*   Czy celem jest wizualizacja danych?
*   Czy celem jest usunięcie szumu i poprawa wydajności modelu?

**Krok 2: Przygotowanie Danych**
*   Wczytaj i oczyść dane.
*   Podziel dane na zbiór treningowy i testowy (`train_test_split`).
*   **Ważne:** Przeskaluj dane (np. za pomocą `StandardScaler`), ponieważ większość algorytmów redukcji wymiarowości (zwłaszcza PCA) jest wrażliwa na skalę cech.

**Krok 3: Wybór Algorytmu (Drzewko decyzyjne)**
*   Czy dane mają strukturę **liniową**?
    *   TAK -> Użyj **PCA**.
        *   Czy zbiór danych jest **ogromny** i nie mieści się w RAM? -> Użyj **Incremental PCA**.
        *   Czy zbiór danych jest duży, ale mieści się w RAM i zależy mi na **szybkości**? -> Rozważ **Randomized PCA**.
*   Czy dane mają strukturę **nieliniową** (zakrzywioną)?
    *   TAK -> Użyj **Manifold Learning**.
        *   Chcesz "rozwinąć" dane? -> Spróbuj **LLE** lub **Isomap**.
        *   Chcesz użyć potężnego, nieliniowego odpowiednika PCA? -> Użyj **Kernel PCA**.
*   Czy głównym celem jest **wizualizacja klastrów**?
    *   TAK -> Użyj **t-SNE** (najlepszy wybór) lub inne algorytmy z `n_components=2` lub `3`.
*   Czy chcesz zredukować wymiary w kontekście **klasyfikacji** (zachowując separację klas)?
    *   TAK -> Użyj **LDA**.

**Krok 4: Ustalenie Docelowej Liczby Wymiarów**
*   Dla **wizualizacji**: ustaw `n_components=2` (dla wykresu 2D) lub `n_components=3` (dla wykresu 3D).
*   Dla **modelu uczenia maszynowego**:
    *   **PCA:** Ustaw `n_components` na ułamek, np. `0.95`, aby zachować 95% wariancji.
    *   **Inne:** Wybierz liczbę wymiarów eksperymentalnie lub narysuj wykres wydajności modelu w zależności od liczby wymiarów (jeśli to możliwe).

**Krok 5: Implementacja i Trenowanie**
*   Stwórz instancję wybranego algorytmu redukcji.
*   Użyj `fit_transform()` na **zbiorze treningowym**.
*   Użyj **tylko `transform()`** na **zbiorze testowym**.

**Krok 6: Ewaluacja Wyników**
*   **Podejście pośrednie (dla modelu):**
    1.  Wytrenuj model (np. klasyfikator) na danych **oryginalnych** i zmierz jego wydajność i czas treningu.
    2.  Wytrenuj ten sam model na danych **zredukowanych** i porównaj wydajność i czas. Czy trening był szybszy? Czy dokładność spadła nieznacznie, czy może nawet wzrosła?
*   **Podejście bezpośrednie (dla PCA/kPCA):**
    *   Oblicz **błąd rekonstrukcji**, aby sprawdzić, ile informacji zostało utracone.

**Krok 7: Dostrajanie Hiperparametrów (jeśli potrzeba)**
*   Dla **kPCA** lub **LLE**, użyj `GridSearchCV` w połączeniu z `Pipeline`, aby znaleźć najlepsze hiperparametry (np. `kernel`, `gamma`, `n_neighbors`) pod kątem finalnego zadania.

**Krok 8: Finalizacja**
*   Zapisz wytrenowany obiekt redukcji wymiarowości (np. `pca`) oraz finalny model, aby móc ich używać na nowych, niewidzianych wcześniej danych. Pamiętaj o zachowaniu tej samej sekwencji operacji: skalowanie -> redukcja wymiarowości -> predykcja.