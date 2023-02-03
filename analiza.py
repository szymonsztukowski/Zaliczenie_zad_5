# importowanie bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###############################
# ZMIENNE, FUNKCJE, OBLICZENIA
###############################

# wczytywanie danych z pliku CSV
df = pd.read_csv('dane/train.csv')
# usunięcie wadliwych danych
data = df.drop(['Cabin', 'Ticket', 'Name', 'Fare', 'PassengerId'], axis=1)
# modyfikacja danych dla lepszej czytelności
data.loc[:, 'Embarked'].replace(['C', 'S', 'Q'], ['Cherbourg', 'Southampton', 'Queenstown'], inplace = True)
# Grupowanie danych według kolumn do dalszej analizy, liczenie średniej
survival_rate = data.groupby(['Sex', 'Pclass', 'Age'])['Survived'].mean()
# Zgromadzenie wszystkich zmiennych
zmienne = list(data)
# Obliczanie całkowitej liczby kobiet i mężczyzn na pokładzie
liczba_mezczyzn = data.Sex.value_counts().male
liczba_kobiet = data.Sex.value_counts().female
# Podzielenie pasażerów na rodziny i osoby samotne
# Dodanie kolumny z rozmiarem rodziny
data['Rozmiar_rodziny'] = 0
data['Rozmiar_rodziny'] = data['Parch'] + data['SibSp']

def histogram_wieku():
    
    ''' Funkcja wyświetla histogram wieku ofiar z podanego zbioru danych. '''

    plt.hist(data['Age'].dropna(), bins=20, width=3)
    plt.xlabel('Wiek')
    plt.ylabel('Liczba ofiar')
    plt.title('Histogram wieku ofiar')
    plt.show()

def wykres_plec_smierc():

    ''' Funkcja wyświetla wykres słupkowy, który wskazuje na liczbę ofiar według płci. '''

    plec_smierc = sns.catplot(x ="Sex", hue = "Survived", kind = "count", data = data, legend = False)
    plt.xlabel("Płeć")
    plt.ylabel("Liczba")
    plt.gca().legend(('Martwi','Żywi'))
    plec_smierc.set_xticklabels(['Mężczyźni', 'Kobiety'])
    plt.show()

def wykres_port_miejsce():

    ''' Funkcja wyświetla wykres słupkowy, który wskazuje liczbę pasażerów z danego portu oraz klasę miejsca, jakie wykupili. '''

    port_klasa = sns.catplot(x = "Embarked", hue = "Pclass", kind = "count", data = data, legend = False)
    plt.xlabel("Port początkowy")
    plt.ylabel("Liczba")
    plt.gca().legend(title = "Klasa miejsca", labels = ['1', '2', '3'])
    plt.show()

def wykres_port_plec():

    ''' Funkcja wyświetla wykres słupkowy, który wskazuje liczbę pasażerów z danego portu w zależności od płci. '''

    port_plec = sns.catplot(x = "Embarked", hue = "Sex", kind = "count", data = data, legend = False)
    plt.xlabel("Port początkowy")
    plt.ylabel("Liczba")
    plt.gca().legend(title = "Płeć", labels = ['Mężczyźni', 'Kobiety'])
    plt.show()

def wykres_port_przetrwanie():

    ''' Funkja wyświetla wykres słupkowy, który wskazuje liczbę ofiar w zależności od portu początkowego. '''

    port_przetrwanie = sns.catplot(x = "Embarked", hue = "Survived", kind = "count", data = data, legend = False)
    plt.xlabel("Port początkowy")
    plt.ylabel("Liczba")
    plt.gca().legend(('Martwi','Żywi'))
    plt.show()

def mapa_cieplna():

    ''' Funkcja grupuje ze sobą kolumny klasy miejsca (Pclass) oraz ofiar (Survived), następnie sprawdza rozmiar tej grupy oraz tworzy mapę cieplną, która wskazuje zależności między klasą miejsca a stopniem przeżywalności. '''

    # Grupowanie zmiennych według przeżywalności oraz klasy miejsca
    grupa_klasa_zycie = data.groupby(['Pclass', 'Survived'])
    klasa_zycie = grupa_klasa_zycie.size().unstack()
 
    # Tworzenie mapy cieplnej wskazującej zależności między przeżywalnością a klasą miejsca (im mniejsza klasa - większy koszt biletu)
    mapa_cieplna = sns.heatmap(klasa_zycie, annot = True, fmt ="d")
    plt.xlabel(None)
    plt.ylabel("Klasa miejsca")
    mapa_cieplna.set_xticklabels(['Martwi', 'Żywi'])
    plt.show()

def wykres_klasa_wiek():

    ''' Funkcja tworzy wykres pudełkowy, który wskazuje na zależności między wiekiem a klasą miejsca. '''

    sns.boxplot(x = 'Pclass', y = 'Age',data = data)
    plt.xlabel("Klasa miejsca")
    plt.ylabel("Wiek")
    plt.show()

def wykres_wioliniowy():

    ''' Funkcja tworzy wykres wioliniowy, który dzieli zbiór na kobiety i mężczyzn, wskazuje przeżywalność oraz wiek pasażerów. '''

    rozklad = sns.violinplot(x = "Sex", y = "Age", hue = "Survived", data = data, split = True, legend = True, palette = "hls")
    plt.xlabel("Płeć")
    plt.ylabel("Wiek")
    rozklad.set_xticklabels(['Mężczyźni', 'Kobiety'])
    # legenda
    import matplotlib.patches as mpatches
    martwi_patch = mpatches.Patch(color='#CB6D67', label='Martwi')
    zywi_patch = mpatches.Patch(color='#67C5CB', label = 'Żywi')
    rozklad.legend(handles = [martwi_patch, zywi_patch], labels = ['Martwi','Żywi'], ncol = 2)
    plt.show()

def wykres_rodzin():

     ''' Funkcja tworzy wykres, który dzieli zbiór na liczbę członków rodziny poszczególnych pasażerów. '''
    
     wykres_rodzin = sns.catplot(x ='Rozmiar_rodziny', y ='Survived', data = data, kind = "point")
     plt.xlabel('Rozmiar rodziny')
     plt.ylabel('Odsetek przeżywalności')
     plt.show()

###############################
# MENU CLI
###############################
while True:
    print("Witaj w programie do analizy danych katastrofy Titanica!")
    print("Wpisz A, by wyświetlić przykładowe dane z tabeli.")
    print("Wpisz B, by wyświetlić histogram wieku ofiar.")
    print("Wpisz C, by dokonać pełnej analizy danych.")
    print("Wpisz D, by wyświetlić rozkład płci.")
    print("Wpisz E, by wyświetlić rozkład pasażerów ze względu na port.")
    print("Wpisz F, by wyświetlić zależność klasy miejsca i liczby zgonów. ")
    print("Wpisz G, by wyświetlić zależność płci i wieku w zakresie przeżywalności")

    wybor = input("\nWpisz literę przypisaną do jednej z powyższych opcji: ")

    ###############################
    # LOGIKA MENU CLI
    ###############################
    if wybor.upper() == "A":
        print(data.head())
    elif wybor.upper() == "B":
        histogram_wieku()
    elif wybor.upper() == "C":
        print("\nAnaliza danych ofiar Titanica.")
        print("\nNa początek zapoznajmy się z podstawowymi danymi.")
        print(data.head())
        print(f"\nMożemy zauważyć następujące zmienne: {zmienne}. {(zmienne)[0]} wskazuje na to, czy pasażer przeżył (0 - nie, 1 - tak). Zmienna {zmienne[1]} wskazuje klasę miejsca - im niższa klasa, tym więcej pasażer zapłacił za bilet. {zmienne[2]} mówi nam o tym, jakiej płci był pasażer. {zmienne[3]} to wiek pasażera. {zmienne[4]} wskazuje, czy pasażer miał na pokładzie kogoś z rodziny (np. brat lub siostra). {zmienne[5]} określa liczbę dzieci pasażera na pokładzie.")
        print("\nZnając opis zmiennych, możemy teraz wyświetlić podstawowe obliczenia naszego zbioru danych - wystąpienia, średnią, odchylenie standardowe, wartość minimalną, wartość maksymalną, a także kwartyle (25%, 50%, 75%).\n")
        print(data.describe())
        print("\n############################")
        print("\nCo mogło mieć wpływ na szanse na przeżycie pasażerów?\n")
        survival_rate = data.groupby(['Sex', 'Pclass', 'Age'])['Survived'].mean()
        print(survival_rate)
        print("\nJak powszechnie wiadomo - to kobiety i dzieci ratuje się w pierwszej kolejności. Przyjrzyjmy się statystyce przeżywalności względem płci.")
        print("Wykres pokazuje, że śmierć poniosło zdecydowanie więcej mężczyzn, a ponad dwie trzecie kobiet zostało uratowanych. Teoria, że w pierwszej kolejności ratuje się kobiety ma tu jasne zastosowanie.")
        wykres_plec_smierc()
        print("\nPrzyjrzyjmy się teraz temu, czy pasażerowie wsiadający w konkretnych portach mogli mieć nieco więcej szczęścia od pozostałych.")
        print("Według danych - najwięcej osób z miejsc 1. klasy wsiadło w Southampton i Cherbourg. Z Southampton Titanic zabrał również najwięcej osób, co wskazuje także dominująca wartość klasy trzeciej nad pozostałymi portami. W Queenstown pasażerowie wybierali prawie tylko najtańszy bilet, natomiast w Cherbourg wsiadały nieco bardziej zamożne osoby.")
        wykres_port_miejsce()
        print(f"\nW każdym z portów to mężczyźni stanowili największą grupę pasażerów. Łącznie było ich {liczba_mezczyzn}. Kobiet natomiast było łącznie {liczba_kobiet}.")
        wykres_port_plec()
        print(f"\nOczywiście to z portu liczącego najwięcej pasażerów otrzymujemy największą liczbę ofiar i ocalonych.")
        wykres_port_przetrwanie()
        print(f"\nZobaczmy teraz mapę cieplną, która wskazuje zależności między liczbą zgonów i ocalonych a klasą miejsca, czyli zamożnością pasażerów.")
        print(f"\nZ wizualizacji jasno wynika, że największą grupę ocalonych stanowią osoby z pierwszej klasy, czyli te najbardziej zamożne. Najwięcej zgonów jest w klasie trzeciej - to osoby mniej majętne, które stanowiły także największą liczbę pasażerów.")
        mapa_cieplna()
        print(f"\nCzy wiek osób na pokładzie Titanica miał wiele wspólnego z szansą na przeżycie?")
        print(f"Jak najbardziej. Kolejny wykres przedstawia, że w pierwszej klasie znajdowały się osoby głównie w wieku 30-50 lat, a niektóre z nich miały nawet 80 lat. W klasie drugiej i trzeciej największy odsetek osób to pasażerowie w wieku 20-35. Możemy śmiało założyć, że klasą pierwszą podróżowały osoby starsze i w konsekwencji bardziej zamożne, dlatego szansa na ich ocalenie była znacznie wyższa, gdyż miały pierwszeństwo podczas wsiadania do łodzi ratunkowych.")
        wykres_klasa_wiek()
        print(f"\nMożemy jeszcze bardziej wyeksplorować te dane, badajać dokładny rozkład martwych i ocalonych na podstawie wieku i płci.")
        print(f"Wykres wiolinowy jasno wskazuje, że największą liczbę zgonów możemy zauważyć w grupie mężczyzn w wieku 20-30 lat. Kobiety w przedziale wiekowym 20-40 miały wówczas największe szanse na przeżycie.")
        wykres_wioliniowy()
        print(f"\nMożemy także zauważyć, że rodziny liczące 3 osoby najczęściej przeżywały katastrofę, a osoby samotne najczęściej ginęły. Może mieć to związek z tym, że rodzina składała się najczęściej z kobiety, mężczyzny i dziecka - wówczas mężczyźni dbają o to, by członkowie ich rodzin otrzymali miejsce w łodzi ratunkowej, a następnie ci członkowie starają się utrzymać przy życiu mężów i ojców. Osoby samotne - głównie mężczyźni - były najczęściej zdane na siebie. Liczba 0 na naszym wykresie stanowi osoby samotne, a liczba 1 to pary.")
        wykres_rodzin()
        print(f"\nMożemy zatem stwierdzić, że realny wpływ na przeżywalność pasażerów mają następujące czynniki: płeć, wiek, zamożność, liczba członków rodziny na pokładzie. Osoby starsze, kobiety, osoby zamożne oraz te, które posiadają członków swoich rodzin na statku, mają zdecydowanie większe szanse na przeżycie niż mężczyźni w wieku 20-40 lat, osoby mniej zamożne oraz samotne.")
    elif wybor.upper() == "D":
        print("\nPrzedstawiam wykres pasażerów z podziałem na płeć.")
        wykres_plec_smierc()
    elif wybor.upper() == "E":
        print("\nPrzedstawiam dane dotyczące portów.")
        wykres_port_miejsce(), wykres_port_plec(), wykres_port_przetrwanie()
    elif wybor.upper() == "F":
        print("\nPrzedstawiam mape cieplną z zależnością klasy miejsca i zgonów.")
        mapa_cieplna()
    elif wybor.upper() == "G":
        print("\nPrzedstawiam wykres z zależnością płci, wieku i przezywalności")
        wykres_wioliniowy()
    else:
        print("\nNiepoprawna wartość!")

    kontynuacja = input("\n Czy chcesz wrócić do menu? (T/N)")
    
    if kontynuacja.upper() == "T":
        continue
    elif kontynuacja.upper() == "N":
        print("\nDziękuję za skorzystanie z mojego programu!")
        break
    else:
        print("\nNiepoprawna decyzja, wyłączam program.")
        break