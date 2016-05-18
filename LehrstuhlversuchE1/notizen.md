# Ansteuerung eines Powersupplys

Bedienungsanleitung  http://shop.elektroautomatik.de/shop/dos2/Web/manuale/33200260_DE_EN.pdf  

Angesteuert werden soll das PS 9080 2U der Firma Elektro Automatik.
Das Gerät verfügt über zwei Drehregler, einen Touchscreen, CAN Bus,  einen USB Anschluss und ein Ethernet interface zur externen Steuerung.
In dem Netzteil ist außerdem ein Funktionsgenerator integriert.

Dem Netzteil soll eine Startspannung, eine Endspannung und eine Zeit gegeben werden
so dass, das Netzteil im gegebene Zeitraum linear die Spannung von der Start- zur
Endspannung verändert. Kurzgesagt eine Rampingfunktion soll implementiert werden.

Über den rückseitigen USB Anschluss wird das Powersupply an einen Rechner angeschlossen. Dazu wird unter Windows ein Treiber benötigt
welcher über die Website des Hersteller zur Verfügung gestellt wird. http://www.elektroautomatik.de/de/downloads.html
Das Netzteil unterstützt eine Reihe von Kommunikationsprotokollen.

  1. CAN
  2. Profibus
  3. Modbus
  4. RS323
  5. DeviceNet

Das GBIP Protokoll, nach IEEE 488,  wird nicht unterstützt.
Über den USB Port kann das Gerät über SCPI Befehle gesteuert werden. Eine Auflistung
aller unterstützen SCPI Befehle findet sich unter http://www.elektroautomatik.de/de/interfaces-ifab.html (Klick auf Programmieranleitungs-Paket)

Mithilfe von SCPI kann man das Netzteil sehr einfach über z.B. einige Python Befehle bedienen um Spannungen und Ströme einzustellen.
Dazu kann die sehr einfach zu bedienende PyVisa libary benutzt werden http://pyvisa.readthedocs.io/en/stable/.

Für die einfache integration in Laboraufbauten mit anderen Steuerbaren Geräten, soll das Gerät mit einem Labview Programm bedient werden.
Der Hersteller liefert dazu SubViews, LabView subroutinen, mit denen das Netzteil gesteuert werden kann.  Außerdem gibt es vom Hersteller einige Beispielprogramme
mit denen unter anderem der interne Funktionsgenerator des PS 9080 benutzt werden kann.



## Probleme mit LabView
Uns wurde im Verlauf der Labview Programmierung nicht klar wie man
sicheres Multithreading benutzt. Das vor allem aus zwei Gründen:

  1.  Der UI-Thread scheint zu blockieren sobald irgendwo eine verschachtelte
  Struktur in einem Parallelen Kontrollfluss auftaucht. Dadurch ist es uns nicht
  gelungen ein laufendes Program sicher über die Grafische Oberfläche zu beenden
  und das Powersupply zu stoppen.

  2.  Wir haben keine Möglichkeit gefunden um Steuerbefehle an das Netzteil per
  Queue zu versenden. Da das Netzteil bei gleichzeitigen Senden von mehreren Steuerbefehlen abstürtzt muss mit aufwändigen Mutex Strukturen und Locks im Program gearbeitet werden um die Ressource des Netzteil global für alle Threads zu Sperren.

Die Probleme lassen sich weitgehend umgehen indem man den eingebauten Funktionsgenerator nutzt. Dieser unterstützt von Haus aus die Möglichkeit einer Rampingfunktion.

## Probleme mit dem Gerät
Ein weiterer USB-Port befindet sich auf der Vorderseite des Gehäuses über welche
XY Daten für den Funktionsgenerator geladen werden können. Desweiteren kann über
diesen Anschluss die Firmware aktualisiert werden. Dies konnte allerdings nicht geschehen da der Touchscreen defekt ist.

Das Gerät war außerdem extrem instabil und stürtzte häufiger komplett ab und konnte nur durch einen Neustart wieder bentutzt werden.  
