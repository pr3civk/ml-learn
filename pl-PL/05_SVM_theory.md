### **Rozdział 5: Maszyny Wektorów Nośnych (SVM)**

#### **Wprowadzenie**
SVM to uniwersalny model uczenia maszynowego. Można go używać do:
*   **Klasyfikacji:** Dzielenia danych na grupy (np. spam vs. nie-spam).
*   **Regresji:** Przewidywania wartości liczbowej (np. ceny domu).
*   **Wykrywania anomalii:** Znajdowania nietypowych danych (np. oszustwa na karcie kredytowej).

Jest szczególnie skuteczny przy zbiorach danych, które nie są ogromne, ale są złożone.

---

### **Podrozdział: Liniowa klasyfikacja SVM**

#### **Teoria i Przeznaczenie**

Główną ideą SVM jest znalezienie "najlepszej" linii (lub płaszczyzny w wielu wymiarach), która rozdziela dane na dwie klasy. Ale co to znaczy "najlepsza"?

*   **Klasyfikacja z dużym marginesem (Large Margin Classification):** Wyobraź sobie, że między dwiema grupami danych (np. kropkami niebieskimi i żółtymi) chcesz narysować ulicę. Możesz narysować ją bardzo blisko jednej grupy lub drugiej. SVM stara się narysować tę ulicę tak, aby była **jak najszersza**, a linia środkowa (decyzyjna) była jak najdalej od najbliższych punktów z obu grup. Ta "ulica" to **margines**.
    *   **Dlaczego to robimy?** Szerszy margines oznacza, że model jest bardziej "pewny" swojej decyzji. Daje to większą szansę, że nowe, nieznane dane zostaną poprawnie sklasyfikowane. Model z wąskim marginesem jest jak strzelec, który ledwo trafia w tarczę – mała zmiana i już chybi. Model z szerokim marginesem trafia w sam środek – jest bardziej odporny na drobne wahania w danych i lepiej generalizuje.

*   **Wektory nośne (Support Vectors):** To są te punkty danych, które leżą na krawędzi "ulicy" (marginesu). Są to najważniejsze punkty w całym zbiorze treningowym.
    *   **Dlaczego są ważne?** Tylko te punkty decydują o tym, gdzie przebiega ulica. Gdybyś usunął jakikolwiek inny punkt, który jest daleko od granicy, ulica się nie zmieni. Ale jeśli przesuniesz wektor nośny, cała ulica (i granica decyzyjna) będzie musiała się dostosować. To one "podtrzymują" (ang. *support*) całą konstrukcję.


*Na obrazku po prawej, linia ciągła to granica decyzyjna, a linie przerywane to krawędzie "ulicy" (marginesu). Punkty na krawędziach to wektory nośne.*

*   **Wrażliwość na skalę cech (Sensitivity to feature scales):**
    *   **Problem:** Jeśli jedna cecha ma duży zakres wartości (np. od 0 do 1000), a druga mały (np. od 0 do 1), SVM będzie faworyzować tę cechę o mniejszym zakresie. Ulica będzie bardzo "ściśnięta" wzdłuż osi z dużymi wartościami.
    *   **Rozwiązanie:** **Skalowanie cech**. Musimy sprowadzić wszystkie cechy do porównywalnego zakresu, np. używając `StandardScaler`. To tak, jakbyśmy patrzyli na mapę, gdzie 1 cm w pionie i 1 cm w poziomie oznaczają tę samą odległość. Bez tego SVM nie znajdzie prawdziwie optymalnej, szerokiej ulicy.

---

### **Podrozdział: Klasyfikacja z miękkim marginesem (Soft Margin Classification)**

#### **Teoria i Przeznaczenie**

*   **Klasyfikacja z twardym marginesem (Hard Margin):** To podejście, w którym bezwzględnie wymagamy, aby wszystkie punkty znalazły się po właściwej stronie ulicy i poza jej krawędzią. Ma dwie wady:
    1.  Działa tylko, gdy dane można idealnie rozdzielić linią.
    2.  Jest ekstremalnie wrażliwy na **outliery** (punkty odstające). Jeden nietypowy punkt może całkowicie zepsuć model, zmuszając go do stworzenia bardzo wąskiej ulicy, która słabo generalizuje.

*   **Klasyfikacja z miękkim marginesem (Soft Margin):** To bardziej elastyczne i realistyczne podejście. Pozwalamy na pewne "naruszenia marginesu" (*margin violations*) – czyli na to, że niektóre punkty znajdą się wewnątrz ulicy, a nawet po jej złej stronie.
    *   **Dlaczego to robimy?** Chcemy znaleźć kompromis między dwoma celami:
        1.  Utrzymaniem jak najszerszej ulicy.
        2.  Ograniczeniem liczby i "wagi" naruszeń marginesu.
    To pozwala modelowi ignorować pojedyncze punkty odstające i skupić się na ogólnym trendzie w danych, co prowadzi do lepszej generalizacji.

*   **Hiperparametr `C`:** To jest "pokrętło" do regulacji tego kompromisu.
    *   **Niskie `C`:** Model bardziej dba o szeroką ulicę (szeroki margines). Pozwala na więcej naruszeń. Prowadzi to do prostszego modelu, który lepiej generalizuje (mniejsze ryzyko przeuczenia).
    *   **Wysokie `C`:** Model bardziej dba o to, by jak najmniej punktów naruszało margines. Ulica może stać się bardzo wąska, aby objąć wszystkie punkty. To zwiększa ryzyko **przeuczenia** (ang. *overfitting*) – model staje się zbyt dopasowany do danych treningowych, włączając w to szum i outliery, i słabo radzi sobie z nowymi danymi.


*Po lewej (niskie `C`=1) margines jest szeroki, ale wiele punktów go narusza. Model prawdopodobnie lepiej uogólnia. Po prawej (wysokie `C`=100) jest mniej naruszeń, ale margines jest węższy i bardziej "poszarpany", co sugeruje przeuczenie.*

#### **Przykład w Pythonie**
```python
# Import potrzebnych bibliotek
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import datasets

# 1. Załadowanie danych
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # Używamy tylko dwóch cech: długość i szerokość płatka
y = (iris["target"] == 2).astype(np.float64)  # Klasyfikujemy, czy to gatunek Iris virginica

# 2. Stworzenie potoku (Pipeline)
svm_clf = Pipeline([
    ("scaler", StandardScaler()),  # Krok 1: Skalowanie cech
    ("linear_svc", LinearSVC(C=1, loss="hinge")), # Krok 2: Trening modelu SVM
])

# 3. Trening modelu
svm_clf.fit(X, y)

# 4. Przewidywanie na nowych danych
svm_clf.predict([[5.5, 1.7]])
```
*   **`Pipeline`**: To narzędzie, które łączy kilka kroków w jeden. Jest bardzo przydatne, ponieważ gwarantuje, że dane zostaną przetworzone w odpowiedniej kolejności (najpierw skalowanie, potem trening). Zapobiega to błędom, takim jak przypadkowe skalowanie danych testowych na podstawie całego zbioru danych.
*   **`StandardScaler()`**: Jak wyjaśniono wcześniej, skaluje dane, aby każda cecha miała średnią 0 i odchylenie standardowe 1. To kluczowe dla dobrego działania SVM.
*   **`LinearSVC(C=1, loss="hinge")`**: To implementacja liniowego SVM.
    *   `C=1`: Ustawiamy kompromis między szerokością marginesu a liczbą naruszeń na umiarkowanym poziomie.
    *   `loss="hinge"`: To matematyczna funkcja, która określa, jak "karane" są punkty naruszające margines.

---

### **Podrozdział: Nieliniowa klasyfikacja SVM**

#### **Teoria i Przeznaczenie**

*   **Problem:** Co jeśli danych nie da się rozdzielić prostą linią?
*   **Rozwiązanie:** Możemy dodać nowe cechy, które są kombinacjami istniejących, aby "przenieść" dane do wyższego wymiaru, w którym staną się liniowo separowalne.
    *   **Przykład:** Wyobraź sobie punkty na jednej linii w kolejności: czerwony, zielony, czerwony. Nie da się ich rozdzielić jednym punktem. Ale jeśli dodamy drugą cechę, np. `x2 = x1^2`, punkty te ułożą się na paraboli i teraz już bez problemu możemy je rozdzielić prostą linią.


*Po lewej dane są nieliniowo separowalne. Po dodaniu nowej cechy (oś Y) stają się liniowo separowalne (można je oddzielić czerwoną linią).*

*   **`PolynomialFeatures`**: Narzędzie w Scikit-Learn, które automatycznie tworzy takie nowe cechy wielomianowe. Np. dla cech `a`, `b` i stopnia 2, stworzy `a^2`, `b^2`, `ab`.

#### **Podrozdział: Kernel wielomianowy (Polynomial Kernel)**

*   **Problem z `PolynomialFeatures`:** Tworzenie nowych cech, zwłaszcza przy wysokim stopniu wielomianu, jest bardzo kosztowne obliczeniowo i zużywa mnóstwo pamięci.
*   **Sztuczka z kernelem (The Kernel Trick):** To matematyczny "trick", który pozwala uzyskać **ten sam wynik**, co po dodaniu cech wielomianowych, ale **bez faktycznego ich tworzenia**. SVM oblicza relacje między punktami tak, jakby były w wyższym wymiarze, operując tylko na oryginalnych danych. To niesamowicie wydajne.
*   **`SVC(kernel="poly")`**: Klasa SVM, która implementuje tę sztuczkę.
    *   `kernel="poly"`: Mówimy modelowi, aby użył sztuczki z kernelem wielomianowym.
    *   `degree`: Stopień wielomianu, który symulujemy.
    *   `coef0`: Dodatkowy hiperparametr do strojenia.

---

### **Podrozdział: Cechy podobieństwa i Kernel RBF (Similarity Features & Gaussian RBF Kernel)**

#### **Teoria i Przeznaczenie**

To kolejne podejście do problemów nieliniowych.

*   **Idea:** Zamiast tworzyć cechy wielomianowe, stwórzmy nowe cechy na podstawie **podobieństwa** każdego punktu do kilku wybranych punktów zwanych **punktami orientacyjnymi (landmarks)**.
*   **Jak to działa?**
    1.  Wybieramy landmark.
    2.  Dla każdego punktu danych obliczamy jego odległość od tego landmarku.
    3.  Używamy funkcji podobieństwa (np. **Gaussowska funkcja radialna bazowa - RBF**), która zwraca 1, jeśli punkt jest w tym samym miejscu co landmark, a bliskie 0, jeśli jest daleko.
    4.  Wynik tej funkcji staje się nową cechą.
*   **Problem:** Jak wybrać landmarki? Najprościej jest uznać **każdy punkt treningowy** za landmark. Ale to by oznaczało, że dla `m` próbek stworzymy `m` nowych cech, co jest niewydajne dla dużych zbiorów.
*   **Rozwiązanie: Kernel RBF.** Znowu z pomocą przychodzi sztuczka z kernelem! Kernel RBF daje ten sam efekt, co metoda z cechami podobieństwa, gdzie każdy punkt jest landmarkiem, ale bez fizycznego tworzenia tych cech. To najpopularniejszy i często najskuteczniejszy kernel.

*   **Hiperparametr `gamma` (γ):** Kontroluje, jak "szeroki" jest wpływ jednego punktu.
    *   **Niskie `gamma`:** Wpływ jest szeroki, każdy punkt oddziałuje na duży obszar. Granica decyzyjna jest bardzo gładka i prosta. Może prowadzić do **niedouczenia** (ang. *underfitting*).
    *   **Wysokie `gamma`:** Wpływ jest wąski, każdy punkt oddziałuje tylko na swoje najbliższe otoczenie. Granica decyzyjna staje się bardzo skomplikowana i "poszarpana", próbując dopasować się do każdego punktu z osobna. Może prowadzić do **przeuczenia** (ang. *overfitting*).


*Widać, jak zmiana `gamma` i `C` wpływa na złożoność granicy decyzyjnej. Małe `gamma` i małe `C` dają prosty model, duże `gamma` i duże `C` dają bardzo złożony model.*

#### **Przykład w Pythonie**
```python
# Używamy klasy SVC, która obsługuje kernele
from sklearn.svm import SVC

# Tworzymy potok z kernelem RBF
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)
```
*   **`SVC(kernel="rbf", ...)`**:
    *   `kernel="rbf"`: Wybieramy sztuczkę z kernelem RBF.
    *   `gamma` i `C`: Dwa najważniejsze hiperparametry do strojenia w nieliniowym SVM. Musimy znaleźć dla nich optymalne wartości, np. za pomocą przeszukiwania siatki (`GridSearchCV`).

---

### **Podrozdział: Złożoność obliczeniowa**

*   **`LinearSVC`**: Bardzo szybki. Jego czas treningu rośnie liniowo wraz z liczbą próbek i cech. Idealny dla dużych zbiorów danych. **Nie wspiera sztuczki z kernelem.**
*   **`SVC`**: Znacznie wolniejszy, zwłaszcza dla dużych zbiorów danych. Czas treningu rośnie kwadratowo lub sześciennie z liczbą próbek. Idealny dla małych i średnich, ale złożonych zbiorów danych. **Wspiera sztuczkę z kernelem.**

---

### **Podrozdział: Regresja SVM (SVR)**

#### **Teoria i Przeznaczenie**

SVM można też użyć do przewidywania wartości liczbowych.

*   **Odwrócenie celu:**
    *   **W klasyfikacji:** Chcemy jak najszerszą ulicę, która ROZDZIELA punkty.
    *   **W regresji:** Chcemy jak najszerszą ulicę, w której zmieści się **jak najwięcej punktów**.
*   **Idea:** Model stara się dopasować linię tak, aby większość danych znalazła się wewnątrz marginesu (ulicy). Karane są tylko punkty, które znajdą się **poza** ulicą.
*   **Hiperparametr `epsilon` (ε):** Kontroluje **szerokość ulicy**. Punkty wewnątrz tej ulicy nie wpływają na błąd modelu. Zwiększając `epsilon`, pozwalamy na większe błędy bez "kary".
*   **Klasy w Scikit-Learn:**
    *   **`LinearSVR`**: Szybka, liniowa regresja SVM.
    *   **`SVR`**: Wolniejsza, ale wspiera kernele do nieliniowej regresji.


*Regresja SVM. Modelowi "nie przeszkadzają" punkty wewnątrz marginesu (ulicy), oznaczonej liniami przerywanymi.*

---

### **Schematyczny Plan Projektu z Użyciem SVM (Przepis)**

Oto uproszczona instrukcja krok po kroku, jak podejść do problemu klasyfikacji z użyciem SVM.

**1. Przygotowanie Danych**
    *   Załaduj dane.
    *   Zbadaj i zwizualizuj dane, aby zrozumieć ich strukturę.
    *   Podziel dane na zbiór treningowy i testowy (`train_test_split`). To kluczowe, aby móc obiektywnie ocenić model na danych, których nie widział podczas treningu.

**2. Przetwarzanie Wstępne (Preprocessing)**
    *   Utwórz potok (`Pipeline`).
    *   W potoku umieść `StandardScaler`. Skalowanie cech jest praktycznie obowiązkowe dla SVM.

**3. Wybór i Trening Modelu Bazowego**
    *   Zawsze zaczynaj od najprostszego modelu. W potoku dodaj `LinearSVC`. Jest szybki i stanowi świetny punkt odniesienia.
    *   Wytrenuj cały potok na zbiorze treningowym (`pipeline.fit(X_train, y_train)`).

**4. Ocena Modelu Bazowego**
    *   Oceń wydajność modelu na zbiorze testowym (`pipeline.score(X_test, y_test)`).
    *   Sprawdź metryki (np. dokładność, precyzja, czułość).

**5. Eksperymenty z Modelem Nieliniowym (jeśli model liniowy jest za słaby)**
    *   Zmodyfikuj potok, zastępując `LinearSVC` przez `SVC(kernel="rbf")`. Kernel RBF jest potężnym i uniwersalnym wyborem.
    *   Wytrenuj i oceń ten nowy model.

**6. Strojenie Hiperparametrów**
    *   Jeśli model nieliniowy działa obiecująco, ale nie idealnie, musisz znaleźć najlepsze wartości dla jego "pokręteł" (`C` i `gamma`).
    *   Użyj `GridSearchCV` lub `RandomizedSearchCV`, aby automatycznie przetestować wiele kombinacji hiperparametrów i znaleźć najlepszą z nich na podstawie walidacji krzyżowej.

**7. Ostateczny Trening i Ocena**
    *   Gdy znajdziesz optymalne hiperparametry, stwórz ostateczny model SVM z tymi ustawieniami.
    *   Wytrenuj go na **całym zbiorze treningowym**.
    *   Dokonaj finalnej, ostatecznej oceny na zbiorze testowym, aby oszacować, jak model będzie sobie radził w rzeczywistych warunkach.