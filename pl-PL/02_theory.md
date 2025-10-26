### Wprowadzenie i Wczytanie Danych

Na samym początku projektu musimy wczytać dane. W tym przypadku dane są w pliku CSV (rodzaj pliku tekstowego, gdzie dane są oddzielone przecinkami, coś jak prosty arkusz kalkulacyjny). Używamy do tego biblioteki `pandas`, która jest standardem w analizie danych w Pythonie.

```python
import pandas as pd
import os

# Zmienna HOUSING_PATH to ścieżka do folderu z danymi
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

*   **Cel i przeznaczenie:** Tworzymy małą funkcję, która wczytuje dane. Robimy to w funkcji, aby kod był bardziej uporządkowany. Jeśli w przyszłości będziemy chcieli wczytać dane z innego miejsca lub w inny sposób, wystarczy, że zmienimy tę jedną funkcję, a nie kod w wielu miejscach projektu.
*   **`pandas`:** To biblioteka do Pythona, która ułatwia pracę z danymi tabelarycznymi. Można o niej myśleć jak o super-zaawansowanym Excelu, którym sterujemy za pomocą kodu.
*   **`DataFrame`:** To podstawowa struktura danych w `pandas`. Jest to po prostu tabela z wierszami i kolumnami, bardzo podobna do arkusza w Excelu. Każda kolumna ma swoją nazwę, a wiersze są ponumerowane. W tej tabeli będziemy przechowywać wszystkie nasze dane o nieruchomościach.

---

### Szybki Przegląd Struktury Danych (Take a Quick Look at the Data Structure)

Zanim zaczniemy skomplikowane analizy, musimy "zapoznać się" z naszymi danymi. To tak, jakbyśmy dostali nowy zestaw narzędzi – najpierw chcemy zobaczyć, co jest w środku, zanim zaczniemy budować.

#### Podstawowe metody do inspekcji danych:

1.  **`head()`**
    *   **Czym jest?** Metoda, która pokazuje kilka pierwszych wierszy (domyślnie 5) naszej tabeli (`DataFrame`).
    *   **Po co to robimy?** Nasz zbiór danych może mieć tysiące, a nawet miliony wierszy. Wyświetlanie go w całości byłoby niepraktyczne. `head()` pozwala nam rzucić okiem na dane, zobaczyć, jakie są kolumny, jakiego typu wartości się w nich znajdują (liczby, tekst) i czy na pierwszy rzut oka wszystko wygląda w porządku.
    ```python
    housing = load_housing_data()
    housing.head()
    ```

2.  **`info()`**
    *   **Czym jest?** Metoda, która daje nam techniczne podsumowanie danych.
    *   **Po co to robimy?** `info()` to nasz pierwszy "raport diagnostyczny". Mówi nam:
        *   Ile jest wszystkich wierszy (wpisów). W tym przypadku 20,640.
        *   Jakie są nazwy wszystkich kolumn i jakiego są typu (`float64` to liczba dziesiętna, `object` to zazwyczaj tekst).
        *   **Najważniejsze:** Ile jest wartości niepustych (`non-null`) w każdej kolumnie. W przykładzie widzimy, że kolumna `total_bedrooms` ma tylko 20,433 wartości, podczas gdy inne mają 20,640. To od razu sygnalizuje nam problem: mamy brakujące dane, którymi będziemy musieli się zająć.

3.  **`value_counts()`**
    *   **Czym jest?** Metoda używana na pojedynczej kolumnie (zwanej w `pandas` `Series`), która zlicza, ile razy występuje każda unikalna wartość.
    *   **Po co to robimy?** Jest to niezwykle przydatne dla kolumn, które nie są czysto numeryczne, ale reprezentują kategorie. W naszym przykładzie kolumna `ocean_proximity` (bliskość oceanu) jest typu `object` (tekst). Używając `value_counts()`, dowiadujemy się, jakie są możliwe kategorie (`<1H OCEAN`, `INLAND` itd.) i ile dzielnic należy do każdej z nich. To pomaga nam zrozumieć rozkład tej cechy.
    ```python
    housing["ocean_proximity"].value_counts()
    ```

4.  **`describe()`**
    *   **Czym jest?** Metoda, która generuje statystyczne podsumowanie dla wszystkich kolumn numerycznych.
    *   **Po co to robimy?** Daje nam to szybki wgląd w rozkład liczbowy naszych danych. Widzimy takie wartości jak:
        *   `count`: Liczba wpisów (ponownie, widzimy braki w `total_bedrooms`).
        *   `mean`: Średnia arytmetyczna.
        *   `std`: Odchylenie standardowe (jak bardzo wartości są "rozstrzelone" wokół średniej).
        *   `min` i `max`: Wartość minimalna i maksymalna.
        *   `25%`, `50%`, `75%`: **Percentyle**. Mówią nam, jakie wartości oddzielają kolejne ćwiartki danych. Np. `25%` dla `housing_median_age` wynosi 18, co oznacza, że 25% dzielnic ma domy o medianie wieku poniżej 18 lat. `50%` to inaczej **mediana**.

---

### Utworzenie Zbioru Testowego (Create a Test Set)

To jeden z najważniejszych i często pomijanych kroków w całym procesie. Zanim zaczniemy na dobre analizować dane, musimy odłożyć ich część na bok i obiecać sobie, że na nią nie spojrzymy aż do samego końca.

*   **Zbiór testowy (Test Set):** To fragment naszych danych (zazwyczaj ok. 20%), który będzie służył jako ostateczny, sprawdzian dla naszego modelu. Model w trakcie "nauki" (trenowania) nigdy go nie zobaczy.
*   **Po co to robimy?** Chcemy stworzyć model, który będzie dobrze działał na **nowych, nieznanych danych**, a nie tylko na tych, na których się uczył. Jeśli nie odłożymy zbioru testowego, ryzykujemy, że nieświadomie "dopasujemy" nasz model do całego zbioru danych. To prowadzi do zjawiska zwanego **Data Snooping Bias**.
*   **Data Snooping Bias (Podglądanie danych):** Nasz mózg jest niesamowitą maszyną do wykrywania wzorców. Jeśli będziemy analizować cały zbiór danych (łącznie z testowym), możemy zauważyć jakieś przypadkowe zależności i tak zbudować model, żeby je uwzględniał. W efekcie nasz model będzie świetnie działał na tym konkretnym zbiorze, ale zupełnie zawiedzie w rzeczywistym świecie. To tak, jakby uczeń przed egzaminem dostał wgląd w pytania egzaminacyjne – jego wynik będzie fantastyczny, ale nie będzie odzwierciedlał jego prawdziwej wiedzy. Zbiór testowy to nasz "zapieczętowany egzamin".

#### Sposoby podziału danych:

1.  **Podział losowy**
    *   **Jak to działa?** Po prostu losujemy 20% wierszy i wrzucamy je do zbioru testowego.
    *   **Problem:** Jeśli uruchomimy skrypt ponownie, wylosujemy inny zestaw danych. Z czasem nasz model "zobaczy" cały zbiór, co niweczy cel podziału. Rozwiązaniem jest ustawienie "ziarna" losowości (`random_state`), co gwarantuje, że losowanie zawsze da ten sam wynik.
    ```python
    from sklearn.model_selection import train_test_split
    
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    ```

2.  **Próbkowanie warstwowe (Stratified Sampling)**
    *   **Czym jest?** To mądrzejszy sposób podziału. Zamiast losować całkowicie przypadkowo, dbamy o to, aby zbiór testowy był reprezentatywną miniaturką całego zbioru danych.
    *   **Po co to robimy?** Wyobraźmy sobie, że w naszych danych `median_income` (mediana dochodu) jest kluczową cechą do przewidywania ceny domu. Jeśli przez przypadek do zbioru testowego trafią głównie dzielnice o wysokich dochodach, nasza ocena modelu będzie niemiarodajna. Próbkowanie warstwowe dzieli dane na podgrupy (warstwy), np. na podstawie kategorii dochodu, a następnie losuje odpowiednią liczbę próbek z każdej podgrupy, zachowując te same proporcje co w oryginalnym zbiorze. Dzięki temu mamy pewność, że nasz zbiór testowy nie jest "przekrzywiony" i dobrze odzwierciedla całą populację.
    *   **Różne klasy Stratified Sampling dostępne w scikit-learn:**
        *   **`train_test_split(stratify=...)`**: Najprostsza metoda dla standardowego podziału 80/20 lub podobnego. Używamy parametru `stratify`, który automatycznie dba o zachowanie proporcji kategorii w zbiorze treningowym i testowym.
        *   **`StratifiedShuffleSplit`**: Użyteczna, gdy potrzebujemy większej kontroli lub chcemy wykonać wiele iteracji podziału (np. do Monte Carlo validation). Pozwala na wielokrotne losowe podziały danych.
        *   **`StratifiedKFold`**: Idealna do sprawdzianu krzyżowego (cross-validation). Dzieli dane na k równych części (np. k=5), dbając o to, aby każda część miała takie same proporcje kategorii.
        *   **`StratifiedGroupKFold`**: Specjalna wersja dla danych, gdzie próbki należą do grup (np. pomiary od różnych pacjentów, eksperymenty). Zapewnia, że wszystkie próbki z tej samej grupy trafiają do tego samego folda, co zapobiega tzw. "data leakage".
    *   **Kiedy użyć którą metodę?** Wybierz `train_test_split(stratify=...)` dla większości przypadków - to najprostsze i najbardziej uniwersalne rozwiązanie. Użyj `StratifiedKFold` gdy potrzebujesz cross-validation, `StratifiedShuffleSplit` dla zaawansowanych scenariuszy z wieloma iteracjami, a `StratifiedGroupKFold` tylko gdy masz dane pogrupowane.
    ```python
    # Przykład użycia Stratified Sampling z train_test_split
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Najpierw musimy stworzyć kategorię do warstwowania (stratification)
    # Ponieważ median_income jest wartością ciągłą, dzielimy ją na kategorie
    housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
    
    # Teraz używamy stratified sampling - proporcje kategorii dochodu
    # będą zachowane w obu zbiorach (train i test)
    train_set, test_set = train_test_split(
        housing, 
        test_size=0.2, 
        random_state=42,
        stratify=housing["income_cat"]  # Kluczowy parametr!
    )
    
    # Po podziale możemy usunąć pomocniczą kolumnę income_cat
    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    # Alternatywnie: użycie StratifiedShuffleSplit dla większej kontroli
    from sklearn.model_selection import StratifiedShuffleSplit
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    # StratifiedKFold: idealny do cross-validation
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Iteruje przez 5 foldów (części), każdy zachowuje proporcje kategorii
    for train_index, test_index in skf.split(housing, housing["income_cat"]):
        # train_index zawiera indeksy 4/5 danych (trening)
        # test_index zawiera indeksy 1/5 danych (walidacja)
        train_fold = housing.loc[train_index]
        test_fold = housing.loc[test_index]
        # Tutaj moglibyśmy trenować model i oceniać go na test_fold
        # Powtarzamy to 5 razy, każdy raz z innym podziałem
    
    # StratifiedGroupKFold: dla danych pogrupowanych (np. pomiary od różnych pacjentów)
    from sklearn.model_selection import StratifiedGroupKFold
    
    # Przykładowo: załóżmy, że mamy kolumnę 'patient_id' grupującą pomiary
    # housing["patient_id"] = ...  # Każdy pacjent może mieć wiele rekordów
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    # WAŻNE: wszystkie próbki z tej samej grupy trafiają do tego samego folda
    for train_index, test_index in sgkf.split(housing, housing["income_cat"], 
                                              groups=housing["patient_id"]):
        train_fold = housing.loc[train_index]
        test_fold = housing.loc[test_index]
        # Gwarancja: żaden pacjent nie ma danych w obu zbiorach jednocześnie
    ```

---

### Odkrywanie i Wizualizacja Danych w Celu Uzyskania Wglądu (Discover and Visualize the Data to Gain Insights)

Teraz, pracując **tylko na zbiorze treningowym**, możemy zacząć głębszą analizę. Celem jest zrozumienie zależności w danych, znalezienie ciekawych wzorców i anomalii. Najlepszym narzędziem do tego jest wizualizacja.

#### Wizualizacja Danych Geograficznych

*   **Czym jest?** Ponieważ mamy dane o długości (`longitude`) i szerokości (`latitude`) geograficznej, możemy narysować mapę. Używamy do tego **wykresu rozrzutu (scatterplot)**.
*   **Po co to robimy?** Tabela z liczbami jest abstrakcyjna. Wykres pokazujący punkty na mapie (w tym przypadku Kalifornii) od razu daje nam kontekst. Możemy ulepszyć ten wykres:
    *   `alpha=0.1`: Sprawia, że punkty są półprzezroczyste. Tam, gdzie punkty się nakładają, kolor staje się intensywniejszy. To świetny sposób na pokazanie **gęstości zaludnienia**.
    *   `s=population/100`: Wielkość kółka (`s`) zależy od populacji dzielnicy.
    *   `c=median_house_value`: Kolor kółka (`c`) zależy od mediany ceny domu. Używamy mapy kolorów (`cmap`), gdzie np. niebieski oznacza tanio, a czerwony drogo.
*   **Wnioski:** Taka wizualizacja natychmiast pokazuje, że ceny domów są silnie powiązane z lokalizacją (drożej przy wybrzeżu, w dużych miastach) i gęstością zaludnienia.

#### Szukanie Korelacji (Looking for Correlations)

*   **Korelacja:** To miara statystyczna, która mówi, jak silnie dwie zmienne są ze sobą powiązane liniowo. Współczynnik korelacji (Pearsona) przyjmuje wartości od -1 do 1.
    *   `1`: Idealna korelacja dodatnia (gdy jedna zmienna rośnie, druga też rośnie).
    *   `-1`: Idealna korelacja ujemna (gdy jedna rośnie, druga maleje).
    *   `0`: Brak korelacji liniowej.
*   **Po co to robimy?** Chcemy znaleźć cechy, które najsilniej wpływają na cenę domu (`median_house_value`). Obliczając korelację każdej cechy z ceną, widzimy, że `median_income` ma bardzo silną korelację dodatnią (ok. 0.68). To nasz główny kandydat na najważniejszą cechę.
*   **Ważna uwaga:** Korelacja mierzy tylko **zależności liniowe**. Może całkowicie pominąć bardziej złożone wzorce (np. paraboliczne). Dlatego wizualizacja jest tak ważna.
*   **`scatter_matrix`:** Funkcja `pandas`, która tworzy macierz wykresów. Rysuje wykres rozrzutu dla każdej pary atrybutów. To świetny sposób na wizualne zbadanie korelacji między wieloma zmiennymi naraz.

#### Eksperymentowanie z Kombinacjami Atrybutów

*   **Czym jest?** Proces tworzenia nowych, bardziej informatywnych cech z tych, które już mamy. Nazywa się to **inżynierią cech (feature engineering)**.
*   **Po co to robimy?** Czasem surowe dane nie są najlepszym predyktorem. Na przykład, `total_rooms` (łączna liczba pokoi w dzielnicy) sama w sobie niewiele mówi. Ale jeśli podzielimy ją przez `households` (liczbę gospodarstw domowych), otrzymamy nową cechę `rooms_per_household` (pokoje na gospodarstwo domowe), która znacznie lepiej opisuje, jak duże są domy w okolicy. Podobnie tworzymy `bedrooms_per_room` i `population_per_household`.
*   **Wynik:** Po stworzeniu tych nowych cech i ponownym sprawdzeniu korelacji okazuje się, że np. `bedrooms_per_room` ma silniejszą (ujemną) korelację z ceną niż `total_rooms` czy `total_bedrooms`. Oznacza to, że im mniejszy stosunek sypialni do wszystkich pokoi, tym dom jest droższy. Znaleźliśmy nową, potężną cechę!

---

### Przygotowanie Danych dla Algorytmów Uczenia Maszynowego (Prepare the Data for Machine Learning Algorithms)

Algorytmy ML są jak wybredni szefowie kuchni – potrzebują składników przygotowanych w bardzo konkretny sposób. Ta faza polega na "czyszczeniu" i "formatowaniu" danych. Zamiast robić to ręcznie, piszemy funkcje i **potoki (pipelines)**, aby cały proces był automatyczny i powtarzalny.

#### Czyszczenie Danych (Data Cleaning)

*   **Problem:** W kolumnie `total_bedrooms` brakowało nam danych. Większość algorytmów nie potrafi pracować z brakującymi wartościami.
*   **Rozwiązania:**
    1.  Usunąć wiersze z brakującymi danymi.
    2.  Usunąć całą kolumnę `total_bedrooms`.
    3.  **Wypełnić braki** jakąś wartością (np. zerem, średnią lub medianą).
*   **Najlepsze podejście:** Wypełnienie medianą jest często dobrym wyborem, ponieważ jest mniej wrażliwe na wartości odstające niż średnia.
*   **`SimpleImputer`:** Klasa z biblioteki `scikit-learn`, która automatyzuje ten proces. Tworzymy "imputer", który "uczy się" mediany z danych treningowych, a następnie używamy go do wypełnienia braków.
    *   **Dlaczego `scikit-learn`?** Ta biblioteka zawiera gotowe narzędzia do większości zadań ML. Jej obiekty (jak `SimpleImputer`) mają spójny interfejs: metoda `fit()` do "nauczenia się" czegoś z danych (np. mediany) i metoda `transform()` do zastosowania tej transformacji.

#### Obsługa Atrybutów Tekstowych i Kategorycznych (Handling Text and Categorical Attributes)

*   **Problem:** Algorytmy ML rozumieją tylko liczby. Nasza kolumna `ocean_proximity` zawiera tekst.
*   **Rozwiązania:**
    1.  **`OrdinalEncoder`:** Zamienia każdą kategorię na liczbę (np. `<1H OCEAN` -> 0, `INLAND` -> 1 itd.).
        *   **Kiedy używać?** Tylko gdy kategorie mają naturalną kolejność (np. "zły", "średni", "dobry").
        *   **Dlaczego nie tutaj?** Użycie go na `ocean_proximity` byłoby błędem. Algorytm uznałby, że kategoria 0 i 1 są do siebie "bardziej podobne" niż 0 i 4, co nie ma sensu.
    2.  **`OneHotEncoder` (Kodowanie "gorącojedynkowe")**
        *   **Jak to działa?** Tworzy nową, binarną (0/1) kolumnę dla każdej kategorii. Jeśli dzielnica ma kategorię `INLAND`, to w nowej kolumnie `INLAND` będzie `1`, a w pozostałych (`<1H OCEAN`, `NEAR BAY` itd.) będzie `0`.
        *   **Po co to robimy?** To najlepszy sposób na reprezentowanie danych kategorycznych bez wprowadzania sztucznej kolejności. Algorytm traktuje każdą kategorię jako oddzielną, niezależną cechę.

#### Niestandardowe Transformatory (Custom Transformers)

*   **Czym są?** Możemy tworzyć własne "narzędzia" do transformacji danych, które będą działać tak samo jak te wbudowane w `scikit-learn`.
*   **Po co to robimy?** Wcześniej ręcznie tworzyliśmy nowe cechy (np. `rooms_per_household`). Tworząc własny transformator (np. `CombinedAttributesAdder`), możemy włączyć ten krok do naszego automatycznego potoku przetwarzania danych. To sprawia, że nasz kod jest czysty, modułowy i łatwy do ponownego użycia.

#### Skalowanie Cech (Feature Scaling)

*   **Problem:** Algorytmy ML często działają źle, gdy cechy numeryczne mają bardzo różne skale (np. `total_rooms` w tysiącach, a `median_income` od 0 do 15). Algorytm może niesłusznie przypisać większą wagę cesze o większych wartościach.
*   **Po co to robimy?** Musimy sprowadzić wszystkie cechy do podobnego zakresu.
*   **Metody:**
    1.  **Normalizacja (Min-Max Scaling):** Przeskalowuje wartości tak, aby mieściły się w zakresie od 0 do 1.
    2.  **Standaryzacja (Standardization):** Przesuwa wartości tak, aby miały średnią 0 i odchylenie standardowe 1. Jest mniej wrażliwa na wartości odstające (outliery) i często preferowana.
*   **`StandardScaler`:** Klasa `scikit-learn` do standaryzacji.

#### Potoki Transformacji (Transformation Pipelines)

*   **Problem:** Mamy wiele kroków przetwarzania: wypełnianie braków, dodawanie cech, skalowanie, kodowanie kategorii. Wykonywanie ich po kolei jest kłopotliwe i podatne na błędy.
*   **`Pipeline`:** To obiekt `scikit-learn`, który pozwala połączyć wszystkie te kroki w jedną, spójną sekwencję. Podajemy mu listę transformacji, a on wykonuje je po kolei.
*   **`ColumnTransformer`:** To jeszcze potężniejsze narzędzie. Pozwala zastosować **różne potoki do różnych kolumn**. Na przykład, możemy stworzyć jeden potok dla danych numerycznych (imputer, dodawanie cech, skaler) i drugi dla danych kategorycznych (`OneHotEncoder`), a `ColumnTransformer` połączy wyniki.
*   **Po co to robimy?** Dzięki temu mamy jeden, finalny obiekt transformujący, który przyjmuje surowe dane i zwraca w pełni przygotowane dane, gotowe do podania modelowi. To ogromne uproszczenie i klucz do tworzenia solidnych systemów ML.

```python
# Przykład potoku dla danych numerycznych
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

# A ColumnTransformer połączy to z przetwarzaniem kolumn kategorycznych
# ... i da nam jeden obiekt `full_pipeline`
```

---

### Wybór i Trenowanie Modelu (Select and Train a Model)

Po całym tym przygotowaniu, trenowanie modeli jest już proste.

#### Trenowanie i Ocena na Zbiorze Treningowym

1.  **Model Liniowy (Linear Regression):** Zaczynamy od prostego modelu. Trenujemy go na przygotowanych danych. Okazuje się, że jego błąd (RMSE) jest duży (ok. $68,628). To przykład **niedotrenowania (underfitting)** – model jest zbyt prosty, by nauczyć się złożonych wzorców w danych.
2.  **Drzewo Decyzyjne (Decision Tree Regressor):** Próbujemy potężniejszego, bardziej złożonego modelu. Po treningu błąd na zbiorze treningowym wynosi... 0.0!
    *   **Czy to dobrze?** Absolutnie nie. To klasyczny przykład **przetrenowania (overfitting)**. Model nie "nauczył się" danych, on je "wykuł na pamięć", łącznie z całym szumem i przypadkowymi zależnościami. Taki model będzie fatalnie działał na nowych danych.

#### Lepsza Ocena za pomocą Sprawdzianu Krzyżowego (Cross-Validation)

*   **Problem:** Jak rzetelnie ocenić model, nie dotykając zbioru testowego?
*   **Sprawdzian krzyżowy (Cross-Validation):** To technika, która dzieli zbiór **treningowy** na kilka (np. 10) części (tzw. "folds"). Następnie trenuje model 10 razy. Za każdym razem jedna część jest używana jako zbiór walidacyjny (do oceny), a pozostałe 9 jako treningowy. Na koniec uśredniamy wyniki z 10 przebiegów.
*   **Po co to robimy?** Daje nam to znacznie stabilniejszą i bardziej wiarygodną ocenę wydajności modelu niż pojedynczy podział na zbiór treningowy i walidacyjny. Po zastosowaniu sprawdzianu krzyżowego na Drzewie Decyzyjnym widzimy, że jego błąd jest w rzeczywistości bardzo duży (ok. 71,407), nawet gorszy niż modelu liniowego. To potwierdza, że model mocno się przetrenował.
*   **Model Lasu Losowego (Random Forest):** To model typu **zespołowego (ensemble)**. Zamiast jednego drzewa decyzyjnego, trenuje ich wiele na różnych podzbiorach danych, a następnie uśrednia ich predykcje. Okazuje się, że ten model działa znacznie lepiej, z błędem ok. 50,182. To nasz najlepszy kandydat.

---

### Dopracowanie Modelu (Fine-Tune Your Model)

Mamy już obiecujący model (Las Losowy), ale możemy go jeszcze ulepszyć, dostrajając jego **hiperparametry**.

*   **Hiperparametry:** To "pokrętła" i "suwaki" modelu, które ustawiamy przed treningiem (np. liczba drzew w lesie, maksymalna głębokość każdego drzewa).
*   **Grid Search (Przeszukiwanie siatki):**
    *   **Jak to działa?** Podajemy siatkę możliwych wartości dla różnych hiperparametrów, a `GridSearchCV` automatycznie testuje każdą możliwą kombinację, używając sprawdzianu krzyżowego do oceny.
    *   **Po co to robimy?** To zautomatyzowany sposób na znalezienie najlepszego zestawu hiperparametrów bez żmudnego, ręcznego testowania.
*   **Randomized Search (Przeszukiwanie losowe):**
    *   **Jak to działa?** Gdy siatka kombinacji jest ogromna, `Grid Search` jest zbyt wolny. `Randomized Search` testuje zadaną liczbę losowych kombinacji hiperparametrów.
    *   **Po co to robimy?** Jest to często bardziej efektywny sposób na przeszukiwanie dużej przestrzeni hiperparametrów.
*   **Analiza najlepszych modeli i ich błędów:** Po znalezieniu najlepszego modelu warto sprawdzić, które cechy uznał za najważniejsze (`feature_importances_`). To może dać nam dodatkowy wgląd w problem.

#### Ocena Systemu na Zbiorze Testowym

To jest moment prawdy. Po całym procesie – czyszczeniu, trenowaniu, dostrajaniu – bierzemy nasz finalny, najlepszy model i po raz pierwszy oceniamy go na zbiorze testowym, który do tej pory był nietknięty.
*   **Jak to robimy?** Bierzemy dane testowe, przepuszczamy je przez nasz `full_pipeline` (używając tylko metody `transform()`, a nie `fit_transform()`, bo nie chcemy uczyć się niczego ze zbioru testowego!), a następnie dokonujemy predykcji i obliczamy finalny błąd.
*   **Po co to robimy?** Wynik na zbiorze testowym daje nam najlepsze oszacowanie tego, jak nasz model będzie działał w rzeczywistym świecie na nowych danych.

---

### Uruchomienie, Monitorowanie i Utrzymanie Systemu (Launch, Monitor, and Maintain Your System)

*   **Uruchomienie (Deployment):** Gotowy model (wraz z całym potokiem przetwarzania) trzeba "opakować" tak, aby mógł być używany w praktyce, np. jako część aplikacji webowej.
*   **Monitorowanie:** Model wdrożony na produkcji trzeba stale monitorować. Jego wydajność może z czasem spadać, ponieważ świat się zmienia, a dane, na których był trenowany, stają się nieaktualne. To zjawisko nazywa się **"model rot" (gnicie modelu)**.
*   **Utrzymanie:** System ML wymaga regularnego utrzymania, w tym potencjalnego ponownego trenowania modelu na świeżych danych.

---

### Schematyczny Plan Projektu (Przepis)

Oto przepis na realizację projektu uczenia maszynowego od początku do końca, oparty na powyższym materiale:

1.  **Zrozumienie Problemu:** Zdefiniuj cel biznesowy i sposób, w jaki model będzie używany.
2.  **Pozyskanie Danych:** Zbierz i wczytaj dane do środowiska pracy (np. `pandas DataFrame`).
3.  **Wstępna Eksploracja Danych:** Użyj `head()`, `info()`, `describe()`, `value_counts()`, aby szybko zapoznać się z danymi.
4.  **Utworzenie Zbioru Testowego:** Odetnij ~20% danych i odłóż na bok. Użyj próbkowania warstwowego, jeśli któraś cecha jest szczególnie ważna.
5.  **Głęboka Eksploracja i Wizualizacja (na zbiorze treningowym):**
    *   Twórz histogramy, wykresy rozrzutu.
    *   Szukaj korelacji i wzorców.
    *   Identyfikuj anomalie i wartości odstające.
6.  **Inżynieria Cech:** Eksperymentuj z tworzeniem nowych, bardziej użytecznych cech z istniejących.
7.  **Przygotowanie Danych (Budowa Potoków):**
    *   Stwórz potok dla danych numerycznych (wypełnianie braków, dodawanie cech, skalowanie).
    *   Stwórz potok dla danych kategorycznych (kodowanie, np. One-Hot).
    *   Połącz wszystko w jeden główny transformator (`ColumnTransformer`).
8.  **Wybór i Trenowanie Modeli Bazowych:**
    *   Przetestuj kilka różnych typów modeli (np. liniowy, drzewo decyzyjne, las losowy, SVM).
    *   Użyj sprawdzianu krzyżowego, aby uzyskać wiarygodną ocenę każdego z nich.
9.  **Wybór Najlepszych Modeli:** Na podstawie wyników sprawdzianu krzyżowego, wybierz 2-3 najbardziej obiecujące modele.
10. **Dostrajanie Hiperparametrów:** Użyj `GridSearchCV` lub `RandomizedSearchCV`, aby znaleźć optymalne ustawienia dla wybranych modeli.
11. **Analiza Finalnego Modelu:** Zbadaj najlepszy model – sprawdź ważność cech i przeanalizuj błędy, które popełnia.
12. **Ostateczna Ocena na Zbiorze Testowym:** Użyj nietkniętego dotąd zbioru testowego, aby uzyskać ostateczną, bezstronną ocenę wydajności modelu.
13. **Prezentacja Wyników:** Przygotuj podsumowanie projektu, wnioski i wizualizacje.
14. **Wdrożenie i Monitorowanie:** Uruchom model w środowisku produkcyjnym i stwórz system do monitorowania jego działania w czasie.


--

### Porady Skorpiona (Ostrzeżenia i Pułapki)

Te porady zwracają uwagę na kluczowe, fundamentalne zasady, których zignorowanie może prowadzić do poważnych błędów w projekcie.

#### 1. Konieczność Stworzenia Zbioru Testowego PRZED Analizą

*   **Cytat:** *"Wait! Before you look at the data any further, you need to create a test set, put it aside, and never look at it."*
*   **Kontekst:** Ta porada pojawia się na samym początku, tuż po wczytaniu danych i bardzo pobieżnym rzuceniu na nie okiem.
*   **Wyjaśnienie w Prostym Języku:** To najważniejsza zasada całego procesu. Wyobraź sobie, że przygotowujesz ucznia do bardzo ważnego egzaminu. Dajesz mu materiały do nauki (zbiór treningowy), ale arkusz egzaminacyjny (zbiór testowy) trzymasz pod kluczem. Jeśli uczeń zobaczy pytania egzaminacyjne w trakcie nauki, z pewnością uzyska świetny wynik, ale będzie on fałszywy – nie będzie odzwierciedlał jego prawdziwej wiedzy, a jedynie zdolność do zapamiętania odpowiedzi. Tak samo jest z modelem. Jeśli my, jako twórcy, będziemy analizować cały zbiór danych (włącznie z testowym), nieświadomie "dopasujemy" nasze decyzje i model do specyfiki całego zbioru. Nasza ostateczna ocena będzie zbyt optymistyczna, a model zawiedzie w konfrontacji z prawdziwym światem.
*   **Dlaczego to jest ważne?** Złamanie tej zasady prowadzi do **Data Snooping Bias** (błędu podglądania danych), co unieważnia końcową ocenę modelu. To jak oszukiwanie na egzaminie – wynik jest bezwartościowy.

#### 2. Ograniczenia Współczynnika Korelacji

*   **Cytat:** *"The correlation coefficient only measures linear correlations (“if x goes up, then y generally goes up/down”). It may completely miss out on nonlinear relationships (e.g., “if x is close to 0, then y generally goes up”)."*
*   **Kontekst:** Ta porada pojawia się w sekcji, gdzie analizujemy korelacje między cechami, aby znaleźć te, które najlepiej przewidują cenę domu.
*   **Wyjaśnienie w Prostym Języku:** Współczynnik korelacji jest jak linijka – potrafi mierzyć tylko proste, liniowe zależności. Jeśli dane układają się wzdłuż prostej linii (rosnącej lub malejącej), korelacja to wychwyci. Ale co, jeśli zależność jest bardziej skomplikowana, np. ma kształt litery "U" (parabola)? Wtedy ceny domów są wysokie dla bardzo niskich i bardzo wysokich wartości jakiejś cechy, a niskie pośrodku. Dla takiej zależności współczynnik korelacji może wynieść zero, sugerując brak związku, co jest nieprawdą.
*   **Dlaczego to jest ważne?** Poleganie wyłącznie na liczbowym współczynniku korelacji jest ryzykowne. Możemy przez to przeoczyć ważne, ale nieliniowe zależności. Dlatego zawsze należy **wizualizować dane** (np. za pomocą wykresów rozrzutu), ponieważ nasze oczy są znacznie lepsze w wykrywaniu złożonych wzorców niż prosta metryka statystyczna.

#### 3. Dopasowywanie Transformacji Tylko do Danych Treningowych

*   **Cytat:** *"As with all the transformations, it is important to fit the scalers to the training data only, not to the full dataset (including the test set). Only then can you use them to transform the training set and the test set (and new data)."*
*   **Kontekst:** Ta uwaga pojawia się przy omawianiu skalowania cech (`StandardScaler`).
*   **Wyjaśnienie w Prostym Języku:** Wróćmy do analogii z egzaminem. Skalowanie cech polega na "nauczeniu się" pewnych parametrów z danych, np. średniej i odchylenia standardowego. Te parametry to część "wiedzy", którą model zdobywa. Jeśli do obliczenia tej średniej użyjemy również danych ze zbioru testowego, to tak, jakbyśmy dali uczniowi jakieś wskazówki z arkusza egzaminacyjnego do wykorzystania podczas nauki. Zbiór testowy ma symulować całkowicie nowe, nieznane dane. Dlatego wszelkie "uczenie się" (metoda `fit()`) musi odbywać się **wyłącznie na zbiorze treningowym**. Następnie tę zdobytą "wiedzę" (np. obliczoną średnią) stosujemy do transformacji zarówno zbioru treningowego, jak i testowego (metoda `transform()`).
*   **Dlaczego to jest ważne?** "Skażenie" procesu uczenia danymi z testowego zestawu nazywa się **wyciekiem danych (data leakage)**. Prowadzi to do nierealistycznie dobrych wyników podczas oceny i jest jednym z najczęstszych i najpoważniejszych błędów w uczeniu maszynowym.

#### 4. Funkcja Użyteczności vs Funkcja Kosztu w Sprawdzianie Krzyżowym

*   **Cytat:** *"Scikit-Learn’s cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better), so the scoring function is actually the opposite of the MSE (i.e., a negative value)..."*
*   **Kontekst:** Ta porada wyjaśnia, dlaczego przy ocenie modelu za pomocą sprawdzianu krzyżowego (`cross_val_score`) używamy metryki `"neg_mean_squared_error"` (ujemny błąd średniokwadratowy).
*   **Wyjaśnienie w Prostym Języku:** W uczeniu maszynowym mamy dwa rodzaje metryk. **Funkcje kosztu** (jak błąd) – chcemy, żeby były jak najmniejsze. **Funkcje użyteczności** (jak dokładność) – chcemy, żeby były jak największe. Narzędzia `scikit-learn` do optymalizacji (jak `GridSearchCV` czy `cross_val_score`) są zaprojektowane tak, aby **maksymalizować** wynik. Nie potrafią minimalizować. Jak więc ocenić model za pomocą błędu, który chcemy minimalizować? Stosuje się prostą sztuczkę: zamiast błędu, maksymalizujemy jego **wartość ujemną**. Maksymalizowanie liczby ujemnej jest tym samym co minimalizowanie jej wartości bezwzględnej (np. maksymalizowanie `-10, -20, -50` doprowadzi nas do `-10`, co odpowiada najmniejszemu błędowi).
*   **Dlaczego to jest ważne?** To techniczna, ale kluczowa uwaga. Bez zrozumienia tej konwencji moglibyśmy źle zinterpretować wyniki lub użyć niewłaściwej metryki, co prowadziłoby do wyboru złego modelu.

---

#### 1. Radzenie Sobie z Dużą Liczbą Kategorii

*   **Cytat:** *"If a categorical attribute has a large number of possible categories (e.g., country code, profession, species), then one-hot encoding will result in a large number of input features. This may slow down training and degrade performance. If this happens, you may want to replace the categorical input with useful numerical features..."*
*   **Kontekst:** Ta wskazówka pojawia się po omówieniu kodowania "gorącojedynkowego" (`OneHotEncoder`).
*   **Wyjaśnienie w Prostym Języku:** `OneHotEncoder` jest świetny, ale ma wadę. Jeśli mamy kolumnę z dużą liczbą unikalnych wartości (np. 200 krajów, 1000 zawodów), stworzy on 200 lub 1000 nowych kolumn. To sprawia, że nasze dane stają się ogromne i "rzadkie" (większość wartości to zera). Model może mieć problem z nauką na tak szerokich danych, a trening będzie bardzo wolny. Porada sugeruje mądrzejsze podejście: zamiast tworzyć setki kolumn, spróbujmy zastąpić kategorię kilkoma sensownymi cechami numerycznymi. Na przykład, zamiast kolumny "kraj", możemy dodać kolumny "populacja kraju" i "PKB kraju". Te liczby mogą nieść więcej użytecznej informacji dla modelu.
*   **Dlaczego to jest ważne?** To przykład zaawansowanej **inżynierii cech**. Pokazuje, że nie należy bezmyślnie stosować standardowych technik. Czasem lepsze zrozumienie problemu i danych pozwala na stworzenie znacznie lepszych i bardziej wydajnych cech.

#### 2. Elastyczność `ColumnTransformer`

*   **Cytat:** *"Instead of using a transformer, you can specify the string "drop" if you want the columns to be dropped, or you can specify "passthrough" if you want the columns to be left untouched."*
*   **Kontekst:** Wskazówka dotyczy dodatkowych opcji w `ColumnTransformer`.
*   **Wyjaśnienie w Prostym Języku:** `ColumnTransformer` jest bardzo elastyczny. Oprócz stosowania skomplikowanych potoków do wybranych kolumn, pozwala też na proste operacje. Jeśli chcemy, aby pewne kolumny zostały całkowicie usunięte z finalnego zbioru danych, możemy użyć opcji `"drop"`. Jeśli z kolei chcemy, aby inne kolumny przeszły przez transformator bez żadnych zmian, używamy `"passthrough"`. Domyślnie, wszystkie kolumny, których nie wymienimy, są usuwane, ale można to zmienić.
*   **Dlaczego to jest ważne?** Ta elastyczność sprawia, że `ColumnTransformer` jest kompletnym narzędziem do zarządzania wszystkimi kolumnami w jednym miejscu, co bardzo upraszcza i porządkuje kod przygotowujący dane.

#### 3. Zapisywanie Modeli

*   **Cytat:** *"You should save every model you experiment with so that you can come back easily to any model you want. Make sure you save both the hyperparameters and the trained parameters... You can easily save Scikit-Learn models by using Python’s pickle module or by using the joblib library..."*
*   **Kontekst:** Ta porada pojawia się po wytrenowaniu kilku różnych modeli i porównaniu ich wyników.
*   **Wyjaśnienie w Prostym Języku:** Trening modeli, zwłaszcza tych złożonych, może trwać bardzo długo. Zamiast trenować model od nowa za każdym razem, gdy chcemy go użyć, możemy go zapisać na dysku. Biblioteka `joblib` jest do tego szczególnie polecana, ponieważ jest wydajna przy zapisywaniu dużych obiektów, jakimi często są wytrenowane modele. Zapisany model można potem wczytać w ułamku sekundy i od razu używać do robienia predykcji. Ważne jest, aby zapisywać nie tylko sam model, ale też jego wyniki i ustawienia, aby móc łatwo porównywać różne eksperymenty.
*   **Dlaczego to jest ważne?** To podstawowa zasada organizacji pracy. Oszczędza mnóstwo czasu i pozwala na systematyczne porównywanie i odtwarzanie wyników eksperymentów.

#### 4. Strategia Wyszukiwania Hiperparametrów

*   **Cytat:** *"When you have no idea what value a hyperparameter should have, a simple approach is to try out consecutive powers of 10 (or a smaller number if you want a more fine-grained search...)."*
*   **Kontekst:** Wskazówka dotyczy definiowania siatki parametrów dla `GridSearchCV`.
*   **Wyjaśnienie w Prostym Języku:** Hiperparametry to "pokrętła" modelu. Skąd mamy wiedzieć, jakie wartości testować? Jeśli nie mamy pojęcia, jaka wartość jest dobra, nie ma sensu testować np. `10, 11, 12, 13`. Lepiej jest przetestować wartości o różnych rzędach wielkości, np. `1, 10, 100, 1000`. To pozwala szybko zorientować się, w jakim zakresie leży optymalna wartość. Gdy już znajdziemy ten "z grubsza" najlepszy zakres (np. między 10 a 100), możemy przeprowadzić bardziej szczegółowe poszukiwania w tym węższym przedziale (np. `10, 30, 50`).
*   **Dlaczego to jest ważne?** To praktyczna strategia, która sprawia, że proces dostrajania hiperparametrów jest znacznie bardziej efektywny i nie marnuje czasu na testowanie bardzo podobnych do siebie wartości.

#### 5. Iteracyjne Poprawianie Wyszukiwania Hiperparametrów

*   **Cytat:** *"Since 8 and 30 are the maximum values that were evaluated, you should probably try searching again with higher values; the score may continue to improve."*
*   **Kontekst:** Porada pojawia się po tym, jak `GridSearchCV` znalazł najlepsze parametry, które okazały się być skrajnymi wartościami z podanej siatki.
*   **Wyjaśnienie w Prostym Języku:** Wyobraź sobie, że szukasz najwyższego punktu na jakimś wzgórzu, ale możesz poruszać się tylko w obrębie wyznaczonego kwadratu. Jeśli po przeszukaniu całego kwadratu okaże się, że najwyższy punkt leży na jego krawędzi, jest duża szansa, że prawdziwy szczyt znajduje się poza tym kwadratem. Tak samo jest z `GridSearch`. Jeśli najlepsze znalezione wartości (`max_features=8`, `n_estimators=30`) są maksymalnymi wartościami, jakie pozwoliliśmy mu przetestować, to jest to sygnał, że powinniśmy rozszerzyć zakres poszukiwań i spróbować jeszcze większych wartości.
*   **Dlaczego to jest ważne?** To pokazuje, że dostrajanie hiperparametrów jest procesem iteracyjnym. Wyniki jednego wyszukiwania dają nam wskazówki, jak skonfigurować następne, aby jeszcze bardziej zbliżyć się do optymalnego rozwiązania.

#### 6. Automatyzacja Przygotowania Danych jako Część Wyszukiwania

*   **Cytat:** *"Don't forget that you can treat some of the data preparation steps as hyperparameters. For example, the grid search will automatically find out whether or not to add a feature you were not sure about..."*
*   **Kontekst:** Ostatnia porada w sekcji o dostrajaniu modelu.
*   **Wyjaśnienie w Prostym Języku:** To bardzo potężna idea. Wcześniej stworzyliśmy własny transformator `CombinedAttributesAdder`, który miał hiperparametr `add_bedrooms_per_room` (domyślnie `True`). Oznacza to, że możemy włączyć ten krok przygotowania danych do siatki `GridSearchCV`! Możemy kazać mu przetestować dwie opcje: jedną z dodaną cechą `bedrooms_per_room` i drugą bez niej. `GridSearch` sam, na podstawie wyników, zdecyduje, czy dodanie tej cechy faktycznie poprawia model, czy nie. W ten sposób możemy zautomatyzować nie tylko dostrajanie modelu, ale także wybór najlepszych kroków w przygotowaniu danych.
*   **Dlaczego to jest ważne?** To łączy przygotowanie danych i modelowanie w jeden, spójny proces optymalizacji. Zamiast zgadywać, czy dany krok inżynierii cech jest dobry, pozwalamy, aby dane i wyniki same nam na to odpowiedziały.

---

### Porada Skorpiona: Automatyczne Ponowne Trenowanie Najlepszego Modelu

*   **Cytat:** *"If GridSearchCV is initialized with `refit=True` (which is the default), then once it finds the best estimator using cross-validation, it retrains it on the whole training set. This is usually a good idea, since feeding it more data will likely improve its performance."*
*   **Kontekst:** Ta uwaga znajduje się w sekcji "Dopracowanie Modelu" (`Fine-Tune Your Model`), tuż po tym, jak pokazano, jak uzyskać najlepszy model (`best_estimator_`) z obiektu `GridSearchCV`.
*   **Wyjaśnienie w Prostym Języku:**
    Wyobraź sobie, że jesteś trenerem drużyny sportowej i chcesz znaleźć najlepszy plan treningowy przed wielkim meczem. Masz 100 zawodników (cały zbiór treningowy).
    1.  **Sprawdzian krzyżowy (Cross-Validation):** Zamiast od razu trenować całą drużynę, testujesz różne plany treningowe (różne kombinacje hiperparametrów) na mniejszych, 10-osobowych grupach. Każdy plan jest testowany kilka razy na różnych grupach, aby ocena była sprawiedliwa.
    2.  **Znalezienie najlepszego planu:** Po wszystkich testach dochodzisz do wniosku, że "Plan B" daje najlepsze wyniki.
    3.  **Co teraz?** Czy na wielki mecz wystawisz tylko tę jedną 10-osobową grupę, która testowała "Plan B"? Oczywiście, że nie! Weźmiesz ten zwycięski "Plan B" i zastosujesz go do treningu **całej swojej 100-osobowej drużyny**.

    Dokładnie to robi `GridSearchCV` z opcją `refit=True`. Po przetestowaniu wszystkich kombinacji i znalezieniu tej "najlepszej recepty" (najlepszych hiperparametrów), nie zwraca on po prostu jednego z tych małych modeli testowych. Zamiast tego, bierze tę zwycięską receptę i **automatycznie trenuje zupełnie nowy model od zera, ale tym razem na całym dostępnym zbiorze treningowym**.

*   **Dlaczego to jest ważne?**
    *   **Lepsza Wydajność:** Model wytrenowany na większej ilości danych jest prawie zawsze lepszy i bardziej niezawodny. Ten finalny model, który otrzymujemy, uczył się z większej liczby przykładów niż którykolwiek z modeli używanych podczas etapu testowania (sprawdzianu krzyżowego).
    *   **Wygoda i Automatyzacja:** To ogromne ułatwienie. Nie musisz ręcznie spisywać najlepszych parametrów, tworzyć nowego modelu z tymi parametrami i trenować go samodzielnie. `GridSearchCV` robi to wszystko za Ciebie w jednym kroku. Obiekt `best_estimator_` jest już tym finalnym, w pełni wytrenowanym modelem.
    *   **Gotowość do Użycia:** Dzięki tej opcji, model, który uzyskujesz na końcu procesu dostrajania, jest od razu gotowy do ostatecznej oceny na zbiorze testowym lub do wdrożenia na produkcję. To finalny, najlepszy produkt całego procesu poszukiwaawczego.

Warto zauważyć, że chociaż ta porada jest oznaczona ikoną skorpiona (która zwykle jest ostrzeżeniem), w tym przypadku ma ona charakter informacyjny – wyjaśnia bardzo ważne, domyślne zachowanie narzędzia, które jest kluczowe dla zrozumienia, co tak naprawdę otrzymujemy jako wynik jego pracy.