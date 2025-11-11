Jasne, oto szczegółowe streszczenie i wyjaśnienie pojęć z podanych rozdziałów, przygotowane w sposób ułatwiający naukę.

### **Rozdział 6: Drzewa Decyzyjne**

Drzewa decyzyjne to jedne z najbardziej intuicyjnych algorytmów uczenia maszynowego. Działają na zasadzie zadawania serii pytań "tak/nie" dotyczących cech danych, aby podzielić je na coraz mniejsze grupy, aż do momentu, w którym w każdej grupie znajdą się dane należące do jednej klasy (lub o podobnej wartości w przypadku regresji).

---

#### **1. Szkolenie i Wizualizacja Drzewa Decyzyjnego**

**Teoria i Przeznaczenie:**
Aby zrozumieć, jak działa drzewo decyzyjne, najlepiej jest je zbudować i zwizualizować. Proces ten polega na tym, że algorytm sam znajduje najlepsze pytania (tzw. podziały), które najskuteczniej dzielą dane. Wizualizacja pozwala nam zobaczyć te "pytania" i ścieżki, którymi podążają dane, co sprawia, że model jest bardzo łatwy do interpretacji.

**Przykład w Kodzie:**
Szkolenie klasyfikatora drzewa decyzyjnego na zbiorze danych "Iris" (klasyfikacja gatunków kwiatów).

```python
# Import potrzebnych bibliotek
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Załadowanie danych
iris = load_iris()
# Wybieramy tylko dwie cechy: długość i szerokość płatka (petal length and width)
# Robimy to, aby łatwiej było zwizualizować granice decyzyjne na wykresie 2D
X = iris.data[:, 2:] 
y = iris.target

# Stworzenie i wytrenowanie modelu
# max_depth=2 oznacza, że drzewo może zadać maksymalnie 2 poziomy pytań.
# Jest to forma REGULARYZACJI - zapobiegamy tym, by drzewo stało się zbyt skomplikowane
# i "nauczyło się na pamięć" danych treningowych (przeuczenie/overfitting).
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```

**Pojęcia i Zagadnienia:**
*   **`DecisionTreeClassifier`**: Klasa z biblioteki Scikit-Learn, która implementuje algorytm drzewa decyzyjnego do zadań klasyfikacji. Tworzymy jej obiekt, aby zbudować nasz model.
*   **`max_depth`**: Hiperparametr, który kontroluje maksymalną "głębokość" drzewa, czyli najdłuższą ścieżkę od korzenia do liścia. Jest to kluczowy parametr do **regularyzacji**.
    *   **Dlaczego to robimy?** Niekontrolowane drzewo będzie rosło, aż idealnie dopasuje się do danych treningowych, ucząc się nawet szumu. Ograniczając głębokość, zmuszamy je do nauki tylko najważniejszych, ogólnych wzorców, co sprawia, że lepiej poradzi sobie z nowymi, nieznanymi danymi.
*   **`fit(X, y)`**: Metoda, która trenuje model. "Patrzy" ona na cechy `X` i odpowiadające im etykiety `y`, a następnie buduje strukturę drzewa, która najlepiej dzieli dane.

**Wizualizacja Drzewa:**
Aby zobaczyć, jak wygląda wytrenowane drzewo, używamy `export_graphviz`, który tworzy plik w formacie `.dot`, a następnie konwertujemy go na obraz.

```python
from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file="iris_tree.dot", # Nazwa pliku wyjściowego
    feature_names=iris.feature_names[2:], # Nazwy cech
    class_names=iris.target_names, # Nazwy klas
    rounded=True,
    filled=True
)
# Ten plik .dot można przekonwertować na obraz PNG poleceniem w terminalu:
# $ dot -Tpng iris_tree.dot -o iris_tree.png
```
*   **Przeznaczenie:** Wizualizacja jest kluczowa dla zrozumienia modelu. Pokazuje nam dokładnie, jakie decyzje podejmuje algorytm. Modele, które można tak łatwo zinterpretować, nazywane są modelami **"białej skrzynki" (white box)**.

---

#### **2. Przewidywanie (Making Predictions)**

**Teoria i Przeznaczenie:**
Przewidywanie w drzewie decyzyjnym polega na przejściu przez nie od góry (od korzenia) w dół, odpowiadając na pytania w każdym węźle. Droga kończy się w **liściu (leaf node)**, a klasa przypisana do tego liścia jest ostateczną predykcją.

**Struktura węzła w zwizualizowanym drzewie:**
*   **Pytanie (np. `petal length (cm) <= 2.45`)**: Warunek, który dzieli dane.
*   **`gini`**: **Współczynnik nieczystości Giniego**. Mierzy, jak bardzo wymieszane są klasy w danym węźle.
    *   `gini = 0`: Węzeł jest "czysty" - wszystkie próbki w nim należą do tej samej klasy.
    *   `gini > 0`: Węzeł jest "nieczysty" - zawiera próbki z różnych klas. Algorytm dąży do tworzenia podziałów, które maksymalnie redukują `gini`.
*   **`samples`**: Liczba próbek danych treningowych, które "dotarły" do tego węzła.
*   **`value`**: Pokazuje, ile próbek z każdej klasy znajduje się w tym węźle (np. `[50, 0, 0]` oznacza 50 próbek klasy "setosa" i 0 pozostałych).
*   **`class`**: Klasa dominująca w danym węźle. W liściu jest to ostateczna predykcja.

---

#### **3. Szacowanie Prawdopodobieństw Klas**

**Teoria i Przeznaczenie:**
Drzewo decyzyjne może nie tylko przewidzieć klasę, ale również oszacować prawdopodobieństwo przynależności do każdej z klas. Robi to, obliczając stosunek próbek danej klasy do wszystkich próbek w liściu, do którego trafiła nowa dana.

**Przykład w Kodzie:**
Dla kwiatu o `petal length=5` i `petal width=1.5` drzewo zwróci prawdopodobieństwa.

```python
# Przewiduje prawdopodobieństwa dla każdej klasy
# Dla [5, 1.5] trafia do liścia z value=[0, 49, 5]
# Prawdopodobieństwa: 0/54, 49/54 (~90.7%), 5/54 (~9.3%)
tree_clf.predict_proba([[5, 1.5]]) 
# >> array([[0.        , 0.90740741, 0.09259259]])

# Przewiduje klasę z najwyższym prawdopodobieństwem (w tym przypadku klasa 1 - versicolor)
tree_clf.predict([[5, 1.5]])
# >> array([1])
```

---

#### **4. Algorytm Uczący CART**

**Teoria i Przeznaczenie:**
Scikit-Learn używa algorytmu **CART (Classification and Regression Tree)** do trenowania drzew. Działa on w następujący sposób:
1.  Dzieli zbiór treningowy na dwie części, używając jednej cechy `k` i progu `tk` (np. `długość płatka <= 2.45`).
2.  Wybiera taką parę `(k, tk)`, która tworzy najczystsze podzbiory (minimalizuje **ważoną** nieczystość Giniego).
3.  Proces ten jest powtarzany rekurencyjnie dla każdego podzbioru, aż do osiągnięcia maksymalnej głębokości lub gdy dalszy podział nie zmniejsza już nieczystości.

Jest to algorytm **"zachłanny" (greedy)**, ponieważ na każdym kroku szuka najlepszego możliwego podziału, nie patrząc, czy ta decyzja doprowadzi do optymalnego rozwiązania w przyszłości. Znalezienie globalnie optymalnego drzewa jest problemem NP-zupełnym, dlatego w praktyce używa się heurystyk, takich jak CART.

---

#### **5. Regularyzacja (Regularization Hyperparameters)**

**Teoria i Przeznaczenie:**
Drzewa decyzyjne, jeśli nie zostaną ograniczone, mają naturalną tendencję do **przeuczenia (overfitting)**. Oznacza to, że idealnie dopasowują się do danych treningowych, ale słabo generalizują na nowe dane. Aby temu zapobiec, stosuje się **regularyzację**, czyli nakładanie ograniczeń na model.

**Najważniejsze hiperparametry do regularyzacji:**
*   **`max_depth`**: Maksymalna głębokość drzewa (już omówiona).
*   **`min_samples_split`**: Minimalna liczba próbek, jakie muszą znaleźć się w węźle, aby mógł on zostać podzielony.
*   **`min_samples_leaf`**: Minimalna liczba próbek, jakie muszą znaleźć się w każdym liściu po podziale.
*   **`max_leaf_nodes`**: Maksymalna liczba liści.

**Dlaczego to robimy?** Zwiększając wartości `min_*` lub zmniejszając `max_*`, ograniczamy swobodę modelu. Zmuszamy go do tworzenia prostszych, bardziej ogólnych reguł, co zmniejsza ryzyko przeuczenia. Na przykład, ustawienie `min_samples_leaf=4` (jak na wykresie z danymi "księżyców") tworzy znacznie gładsze i bardziej wiarygodne granice decyzyjne niż model bez ograniczeń.

---

#### **6. Regresja (Regression)**

**Teoria i Przeznaczenie:**
Drzewa decyzyjne mogą być również używane do zadań regresji (przewidywania wartości liczbowych). Działają bardzo podobnie, ale z dwiema kluczowymi różnicami:
1.  Zamiast przewidywać klasę w liściu, przewidują **średnią wartość** wszystkich próbek, które do niego trafiły.
2.  Zamiast minimalizować nieczystość Giniego, algorytm stara się minimalizować **błąd średniokwadratowy (MSE)** podczas tworzenia podziałów.

**Przykład w Kodzie:**
```python
from sklearn.tree import DecisionTreeRegressor

# max_depth=2 to znowu regularyzacja, aby zapobiec przeuczeniu
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
```
*   **`DecisionTreeRegressor`**: Klasa do tworzenia drzew regresyjnych.
*   **Przeuczenie w regresji:** Bez regularyzacji drzewo regresyjne stworzy tyle liści, ile potrzeba, aby każda próbka treningowa miała jak najmniejszy błąd, co prowadzi do "schodkowej", poszarpanej funkcji, która źle generalizuje (jak na lewym wykresie na `Figure 6-6`). Ustawienie `min_samples_leaf=10` wygładza predykcje i tworzy znacznie lepszy model (prawy wykres).

---

#### **7. Niestabilność (Instability)**

**Teoria i Przeznaczenie:**
Główną wadą drzew decyzyjnych jest ich wrażliwość na niewielkie zmiany w danych treningowych. Usunięcie nawet jednej próbki może prowadzić do powstania zupełnie innej struktury drzewa. Są one również wrażliwe na obrót danych, ponieważ tworzą tylko **prostopadłe granice decyzyjne**.

*   **Dlaczego to problem?** Taka niestabilność sprawia, że pojedyncze drzewo może być niewiarygodne. Jest to główny powód, dla którego w praktyce rzadko używa się pojedynczych drzew decyzyjnych, a zamiast tego stosuje się **lasy losowe**, które są zbiorem wielu drzew i łagodzą ten problem.

### **Rozdział 7: Uczenie Zespołowe i Lasy Losowe**

Uczenie zespołowe (Ensemble Learning) polega na łączeniu wielu modeli (nazywanych "uczniami") w jeden potężny model ("zespół"), który zwykle osiąga lepsze wyniki niż jakikolwiek z jego składników. Idea opiera się na "mądrości tłumu" - łącząc różne perspektywy, niwelujemy indywidualne błędy.

---

#### **1. Klasyfikatory Głosujące (Voting Classifiers)**

**Teoria i Przeznaczenie:**
Najprostsza metoda zespołowa. Trenujemy kilka różnych modeli (np. Regresję Logistyczną, SVM, Drzewo Decyzyjne) na tych samych danych. Aby dokonać predykcji, zbieramy "głosy" od każdego z nich i wybieramy klasę, która otrzymała najwięcej głosów.

*   **Głosowanie twarde (Hard Voting)**: Każdy model głosuje na jedną klasę. Wygrywa klasa z największą liczbą głosów.
*   **Głosowanie miękkie (Soft Voting)**: Jeśli modele potrafią szacować prawdopodobieństwa (mają metodę `predict_proba()`), uśredniamy te prawdopodobieństwa dla każdej klasy i wybieramy tę z najwyższym średnim prawdopodobieństwem.
    *   **Dlaczego głosowanie miękkie jest często lepsze?** Ponieważ uwzględnia "pewność siebie" każdego modelu. Głos modelu, który jest bardzo pewny swojej predykcji (np. 99% prawdopodobieństwa), ma większą wagę niż głos modelu, który "zgaduje" (np. 51%).

**Przykład w Kodzie:**
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True) # probability=True jest potrzebne do soft voting

# Tworzymy klasyfikator głosujący
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft' # lub 'hard'
)
voting_clf.fit(X_train, y_train)
```
*   **`VotingClassifier`**: Klasa, która łączy różne klasyfikatory w jeden zespół.
*   **`estimators`**: Lista krotek zawierających nazwę i obiekt każdego modelu w zespole.
*   **`voting`**: Określa typ głosowania.

**Dlaczego to działa?** Działa najlepiej, gdy modele są od siebie **różnorodne** i popełniają **niezależne błędy**. Jeśli jeden model myli się w określonych przypadkach, jest szansa, że inne modele w tych samych przypadkach sobie poradzą, a ich poprawne głosy "przegłosują" ten błędny.

---

#### **2. Bagging i Pasting**

**Teoria i Przeznaczenie:**
Zamiast używać różnych algorytmów, możemy użyć tego samego algorytmu (np. drzewa decyzyjnego) wielokrotnie, ale za każdym razem trenować go na **innym losowym podzbiorze danych treningowych**.

*   **Bagging (Bootstrap Aggregating)**: Losowanie próbek odbywa się **ze zwracaniem**. Oznacza to, że ta sama próbka może zostać wylosowana kilka razy do jednego podzbioru, a inna może nie zostać wylosowana wcale.
*   **Pasting**: Losowanie próbek odbywa się **bez zwracania**. Każda próbka może być wylosowana co najwyżej raz.

**Dlaczego to robimy?** Celem jest stworzenie **różnorodnych modeli**. Mimo że używamy tego samego algorytmu, trenowanie na różnych podzbiorach danych sprawia, że każdy model uczy się nieco innych wzorców. Agregacja ich predykcji (zwykle przez głosowanie) prowadzi do modelu o **niższej wariancji** (mniejszym przeuczeniu) niż pojedynczy model wytrenowany na pełnym zbiorze danych. Bagging zazwyczaj daje nieco lepsze rezultaty, ponieważ losowanie ze zwracaniem wprowadza większą różnorodność.

**Przykład w Kodzie:**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Zespół 500 drzew decyzyjnych, każde trenowane na 100 losowych próbkach (ze zwracaniem)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
)
bag_clf.fit(X_train, y_train)
```
*   **`BaggingClassifier`**: Klasa do tworzenia zespołów metodą bagging/pasting.
*   **`n_estimators`**: Liczba modeli (drzew) w zespole.
*   **`max_samples`**: Liczba próbek losowanych do trenowania każdego modelu.
*   **`bootstrap=True`**: Włącza bagging (losowanie ze zwracaniem). `bootstrap=False` włączyłoby pasting.
*   **`n_jobs=-1`**: Używa wszystkich dostępnych rdzeni procesora do równoległego trenowania modeli. To ogromna zaleta tej metody - jest bardzo dobrze skalowalna.

---

#### **3. Ewaluacja Out-of-Bag (OOB)**

**Teoria i Przeznaczenie:**
W metodzie bagging, dla każdego modelu około 37% próbek z oryginalnego zbioru treningowego nie jest wykorzystywanych do jego trenowania (są to próbki **Out-of-Bag**). Możemy wykorzystać te "odłożone" próbki jako darmowy zbiór walidacyjny do oceny wydajności każdego modelu. Uśredniając wyniki OOB wszystkich modeli, otrzymujemy ocenę całego zespołu bez potrzeby tworzenia oddzielnego zbioru walidacyjnego.

**Przykład w Kodzie:**
```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True # Włączenie oceny OOB
)
bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_) # Wyświetlenie wyniku OOB
```
*   **`oob_score=True`**: Powoduje, że po treningu model automatycznie oblicza swój wynik na próbkach OOB.
*   **`oob_score_`**: Atrybut przechowujący wynik OOB.
*   **Przeznaczenie:** Jest to wygodny sposób na ocenę modelu, szczególnie gdy mamy mało danych i nie chcemy marnować ich na oddzielny zbiór walidacyjny.

---

#### **4. Lasy Losowe (Random Forests)**

**Teoria i Przeznaczenie:**
Las losowy to po prostu zespół drzew decyzyjnych, zazwyczaj trenowany metodą **bagging**. Jest to jeden z najpotężniejszych i najczęściej używanych algorytmów uczenia maszynowego.

Las losowy wprowadza dodatkowy poziom losowości w porównaniu do zwykłego baggingu: podczas budowy każdego drzewa, na etapie szukania najlepszego podziału w węźle, algorytm nie sprawdza wszystkich cech, a jedynie ich **losowy podzbiór**.

*   **Dlaczego to robimy?** Ta dodatkowa losowość jeszcze bardziej **zwiększa różnorodność** drzew w lesie. Jeśli w danych jest jedna bardzo silna cecha, to w zwykłym baggingu większość drzew mogłaby użyć jej jako pierwszego podziału. W lasach losowych, losując podzbiór cech, zmuszamy niektóre drzewa do rozpoczęcia podziału od innych, mniej oczywistych cech. To prowadzi do mniejszej korelacji między drzewami i ostatecznie do silniejszego, lepiej generalizującego modelu (jeszcze niższa wariancja kosztem nieznacznie wyższej obciążalności).

**Przykład w Kodzie:**
```python
from sklearn.ensemble import RandomForestClassifier

# Las losowy z 500 drzewami, gdzie każdy liść może mieć max 16 próbek
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
```
*   **`RandomForestClassifier`**: Zoptymalizowana klasa do tworzenia lasów losowych. Jest wygodniejsza niż `BaggingClassifier(DecisionTreeClassifier(...))`. Posiada wszystkie hiperparametry drzewa decyzyjnego oraz zespołu.

---

#### **5. Ważność Cech (Feature Importance)**

**Teoria i Przeznaczenie:**
Lasy losowe pozwalają w łatwy sposób ocenić, które cechy są najważniejsze dla predykcji. Robią to, mierząc, o ile dana cecha, użyta do podziału w drzewach, średnio redukuje nieczystość (np. Giniego). Scikit-Learn oblicza to automatycznie po treningu.

**Przykład w Kodzie:**
```python
# Po wytrenowaniu lasu na danych Iris
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
```
*   **`feature_importances_`**: Atrybut modelu przechowujący ważność każdej cechy. Suma wszystkich ważności wynosi 1.
*   **Przeznaczenie:** Jest to niezwykle przydatne do zrozumienia danych i selekcji cech. Możemy zobaczyć, które zmienne mają największy wpływ na wynik, co może pomóc w uproszczeniu modelu lub dostarczyć cennych wniosków biznesowych.

---

#### **6. Boosting**

**Teoria i Przeznaczenie:**
Boosting to inna technika zespołowa, która działa **sekwencyjnie**. Zamiast trenować modele równolegle i niezależnie (jak w baggingu), boosting buduje zespół krok po kroku. Każdy kolejny model jest trenowany tak, aby **poprawić błędy swojego poprzednika**.

*   **AdaBoost (Adaptive Boosting)**:
    1.  Trenuje pierwszy prosty model.
    2.  Sprawdza, które próbki treningowe zostały błędnie sklasyfikowane.
    3.  **Zwiększa wagę** tych błędnie sklasyfikowanych próbek.
    4.  Trenuje drugi model, który zwraca większą uwagę na próbki o wyższej wadze.
    5.  Proces jest powtarzany, a każdy kolejny model skupia się na coraz trudniejszych przypadkach.
    *   Ostateczna predykcja to ważona suma głosów wszystkich modeli, gdzie waga zależy od tego, jak dobrze dany model radził sobie na ważonym zbiorze treningowym.

*   **Gradient Boosting**:
    1.  Trenuje pierwszy model.
    2.  Oblicza **błędy (rezydua)**, jakie ten model popełnił na danych treningowych.
    3.  Trenuje drugi model, ale nie na oryginalnych etykietach, lecz na **tych błędach**. Jego zadaniem jest przewidzenie błędów poprzednika.
    4.  Dodaje predykcje nowego modelu do predykcji poprzedniego.
    5.  Oblicza nowe błędy i powtarza proces.
    *   Każdy kolejny model "koryguje" błędy całego dotychczasowego zespołu.

**Wady Boostingu:** Ponieważ proces jest sekwencyjny, **nie da się go łatwo zrównoleglić**, co sprawia, że jest wolniejszy i gorzej skalowalny niż bagging/lasy losowe.

**Przykład w Kodzie (Gradient Boosting):**
```python
from sklearn.ensemble import GradientBoostingRegressor

# n_estimators=3 - zespół 3 drzew
# learning_rate - kontroluje, jak duży wkład ma każde drzewo. Jest to parametr regularyzacyjny.
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)
```
*   **Wczesne zatrzymywanie (Early Stopping):** Kluczowa technika w boostingu. Trenujemy więcej modeli (drzew) niż potrzeba, ale monitorujemy błąd na zbiorze walidacyjnym. Zatrzymujemy trening, gdy błąd przestaje maleć, i wybieramy optymalną liczbę drzew. Zapobiega to przeuczeniu.

---

#### **7. Stacking (Stacked Generalization)**

**Teoria i Przeznaczenie:**
Stacking to bardziej zaawansowana metoda zespołowa. Zamiast używać prostej funkcji (jak głosowanie lub uśrednianie) do agregacji predykcji, trenujemy **nowy model (nazywany blenderem lub meta-uczniem)**, którego zadaniem jest nauczenie się, jak najlepiej łączyć predykcje modeli z niższej warstwy.

**Jak to działa (w uproszczeniu):**
1.  Dzielimy zbiór treningowy na dwie części (np. A i B).
2.  Trenujemy kilka modeli z pierwszej warstwy (np. SVM, Las Losowy) na części A.
3.  Używamy wytrenowanych modeli do zrobienia predykcji na części B.
4.  Te predykcje stają się **nowymi cechami** dla blendera.
5.  Trenujemy blender na tych nowych cechach (predykcjach) i oryginalnych etykietach z części B.

**Dlaczego to robimy?** Blender może nauczyć się skomplikowanych zależności, np. "jeśli model 1 jest bardzo pewny, a model 2 nie, to ufaj modelowi 1", albo "w tym regionie przestrzeni cech, predykcje modelu 3 są bardziej wiarygodne". To pozwala na bardziej inteligentną agregację niż proste głosowanie. Kluczowe jest, aby predykcje używane do trenowania blendera pochodziły z danych, których modele z pierwszej warstwy "nie widziały" (stąd podział na A i B), aby uniknąć wycieku informacji i przeuczenia.

### Schematyczny Plan Projektu (Przepis)

Oto krok-po-kroku plan budowy projektu z wykorzystaniem tych technik, bez szczegółowych opisów i kodu, w formie instrukcji.

1.  **Definicja Problemu:** Zrozum, co chcesz przewidzieć (klasyfikacja czy regresja).
2.  **Przygotowanie Danych:** Zbierz, oczyść i wstępnie przetwórz dane (np. obsługa brakujących wartości).
3.  **Podział Danych:**
    *   Podziel dane na zbiór treningowy i testowy (`train_test_split`). Zbiór testowy odłóż i nie używaj go aż do samego końca.
    *   Jeśli potrzebujesz, wydziel zbiór walidacyjny ze zbioru treningowego.
4.  **Model Bazowy (Baseline):**
    *   Wytrenuj prosty model, np. pojedyncze Drzewo Decyzyjne.
    *   Oceń jego wydajność – to będzie twój punkt odniesienia.
5.  **Strojenie Hiperparametrów Modelu Bazowego:**
    *   Użyj `GridSearchCV` lub `RandomizedSearchCV` do znalezienia optymalnych hiperparametrów (np. `max_depth`, `min_samples_leaf`) dla Drzewa Decyzyjnego, aby zapobiec przeuczeniu.
6.  **Budowa Zespołu (Ensemble):**
    *   **Opcja A (Różne Modele):** Stwórz `VotingClassifier` z kilkoma różnymi, dobrze działającymi modelami (np. Regresja Logistyczna, SVM, Las Losowy).
    *   **Opcja B (Jeden Typ Modelu):** Zbuduj `RandomForestClassifier` lub `BaggingClassifier`. Są to potężne i uniwersalne opcje.
    *   **Opcja C (Sekwencyjnie):** Zbuduj model boostingowy, np. `GradientBoostingClassifier` lub `XGBoost`.
7.  **Strojenie Hiperparametrów Zespołu:**
    *   Ponownie użyj `GridSearchCV`, aby znaleźć optymalne hiperparametry dla całego zespołu (np. `n_estimators`, `learning_rate` dla boostingu, hiperparametry drzew dla lasu).
    *   Dla boostingu zaimplementuj wczesne zatrzymywanie.
8.  **Analiza i Interpretacja (Opcjonalnie):**
    *   Jeśli użyłeś Lasu Losowego, sprawdź ważność cech (`feature_importances_`), aby zrozumieć, co napędza predykcje.
9.  **Ostateczna Ocena:**
    *   Wybierz najlepszy model (pojedynczy lub zespół) na podstawie wyników na zbiorze walidacyjnym (lub walidacji krzyżowej).
    *   Wytrenuj ten ostateczny model na **całym zbiorze treningowym**.
    *   Dokonaj ostatecznej, jednorazowej oceny wydajności na **odłożonym na początku zbiorze testowym**. Ten wynik pokazuje, jak model prawdopodobnie poradzi sobie w rzeczywistych warunkach.
10. **Prezentacja Wyników:** Przedstaw wyniki i wnioski.