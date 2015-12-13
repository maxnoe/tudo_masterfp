#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h>
#endif
#include <Ethernet.h>

// Pin connected to the NeoPixels
#define PIN            7

// Number of NeoPixels in the ring?
#define NUMPIXELS      16

// The builtin LED
#define LED  9


//The url to the webservice. needs to be a char array.
char url[] = "http://verkackte-s1.herokuapp.com";

//Fix mac adress for this device
byte mac[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED};


// initialize the library instance
EthernetClient heroku_client;


//To check connections in a regular intervall we need to save the time since the last connection
unsigned long lastConnectionTime = 0;             // in milliseconds
const unsigned long postingInterval = 60L * 1000L; // delay between updates


//initialize neopixel library. These are the defaults
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

//save webserver response here
String response = "";

//animation speed
const unsigned int delayval = 60; // delay for half a second

void setup() {
    // set mode for internal LED
    pinMode(LED, OUTPUT);
    //ini network adapter
    Ethernet.begin(mac);

    // initialize neo pixels
    pixels.begin();
    pixels.setBrightness(250);
}

void loop() {

  // check whether we should check the server again
  if (millis() - lastConnectionTime > postingInterval) {
    httpRequest();
    delay(1000);
    response = getResponse();
  }
  //response says whehter train is on time or not.
  if(response == "yes"){
        wheel(5, 50, 10, delayval, 16);
  } else if(response == "no"){
        wheel(45, 0, 5, delayval, 16);
  } else if(response == "maybe"){
        wheel(50, 0, 50, delayval, 16);
  } else {
        blink_all_pixel(50, 0 , 0, 100, 5);
  }
}

String getResponse() {
  String result = "";

  while (heroku_client.available()) {
    char c = heroku_client.read();
    result = result + c;
    //last word in the http header.
    if (result.endsWith("vegur")) {
      //remove everything from index 0 to end
      result.remove(0);
      break;
    }
  }
  //now read the content
  while (heroku_client.available()) {
    char c = heroku_client.read();
    result = result + c;
  }
  result.trim();
  return result;
}

// this method connects to the url and makes a http request.
// the requested itself is hardcoded
void httpRequest() {
  // close any connection before send a new request.
  heroku_client.stop();
  rotate(40, 5, 5, 30, 5);
  rotate(40, 5, 5, 30, 5);
  rotate(40, 5, 5, 30, 5);
  // if there's a successful connection:
  if (heroku_client.connect(url, 80)) {

    // send the HTTP GET request:
    heroku_client.println("GET http://verkackte-s1.herokuapp.com/ontime/ HTTP/1.1");
    heroku_client.println("Host: verkackte-s1.herokuapp.com");
    heroku_client.println("User-Agent: arduino-ethernet");
    heroku_client.println("Connection: close");
    heroku_client.println();

    // reset timer.
    lastConnectionTime = millis();
  } else {
    // in case of bad connection. blink red.
    blink_all_pixel(230, 0, 5, 150,  10);
    return;
  }
  rotate(5, 30, 10, 20, 5);
  rotate(5, 30, 10, 20, 5);
  rotate(5, 30, 10, 20, 5);
}

void blink_all_pixel(int r, int g, int b, int delay_value, int repititions){
    for(int r = 0; r < repititions; r++){
        for(int i=0;i<NUMPIXELS;i++){
            pixels.setPixelColor(i, pixels.Color(r,g,b));
        }
        pixels.show();
        delay(delay_value);
        for(int i=0;i<NUMPIXELS;i++){
            pixels.setPixelColor(i, pixels.Color(0,0,0));
        }
        pixels.show();
        delay(delay_value);
    }
}

void rotate(int red, int green, int blue, int delay_value, int num_pix){
        for(int i=0;i<NUMPIXELS;i++){
            for(int n = 0; n < num_pix; n++){
                pixels.setPixelColor((n + i)%NUMPIXELS , red, green, blue);
            }
            pixels.show();
            delay(delay_value);
            for(int n = 0; n < num_pix; n++){
                pixels.setPixelColor((n + i)%NUMPIXELS , 0, 0, 0);
            }
            pixels.show();
            delay(delay_value);
        }
    return;
}


void wheel(int red, int green, int blue, int delay_value, int repititions){

    for(int r = 0; r < repititions; r++){
        for(int i=r;i<NUMPIXELS + r;i++){
            pixels.setPixelColor(i%NUMPIXELS, red, green, blue);
            pixels.show();
            delay(delay_value);
        }

        for(int i=r;i<NUMPIXELS + r;i++){

            pixels.setPixelColor(i%NUMPIXELS, 0, 0, 0);
            pixels.show();
            delay(delay_value);
        }
    }
    return;
}
