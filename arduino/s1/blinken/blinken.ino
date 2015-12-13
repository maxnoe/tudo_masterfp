#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h>
#endif
#include <SPI.h>
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
const unsigned long postingInterval = 30L * 1000L; // delay between updates


//initialize neopixel library. These are the defaults
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

//save webserver response here
String response = "";

//animation speed
const unsigned int delayval = 50; // delay for half a second

void setup() {
    // set mode for internal LED
    pinMode(LED, OUTPUT);
    //ini network adapter
    Ethernet.begin(mac);
    
    // initialize neo pixels
    pixels.setBrightness(250);
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
  
  if(response == "yes"){
        wheel(5, 30, 10, delayval, 16);
  } else if(response == "no"){
        wheel(45, 0, 5, delayval, 16);
  } else {
        wheel(0, 0, 25, delayval, 16);
  }
}

String getResponse() {
  String result = "";

//   while (heroku_client.available()) {
//     char c = heroku_client.read();
//     result = result + c;
//     if (result.endsWith("vegur")) {
//       result.remove(0);
//       break;
//     }
//   }

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

  // if there's a successful connection:
  if (heroku_client.connect(url, 80)) {
    rotate(100, 10, 10, 20, 5);
    rotate(100, 10, 10, 20, 5);
    rotate(100, 10, 10, 20, 5);
    // send the HTTP GET request:
    heroku_client.println("GET http://verkackte-s1.herokuapp.com/ontime/ HTTP/1.1");
    heroku_client.println("Host: verkackte-s1.herokuapp.com");
    heroku_client.println("User-Agent: arduino-ethernet");
    heroku_client.println("Connection: close");
    heroku_client.println();
    //blink for visual feedback. green/blueish color color
    
    // rest timer.
    lastConnectionTime = millis();
  } else {
    // in case of bad connection. blink red.
    blink_all_pixel(30, 0, 5, 150,  10);
  }
    wheel(5, 30, 10, delayval, 2);
    wheel(45, 0, 5, delayval, 2);
    wheel(0, 0, 25, delayval, 2);
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

void rotate(int r, int g, int b, int delay_value, int num_pix){
        for(int i=0;i<NUMPIXELS;i++){
            for(int n = 0; n < num_pix; n++){
                pixels.setPixelColor((n + i)%NUMPIXELS , r, g, b);     
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


void wheel(int r, int g, int b, int delay_value, int repititions){
    for(int r = 0; r < repititions; r++){
        for(int i=r;i<NUMPIXELS + r;i++){
            pixels.setPixelColor(i%NUMPIXELS, r, g, b); 
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



