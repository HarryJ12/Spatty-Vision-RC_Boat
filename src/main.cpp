/*
 * LAB Name: Arduino OV7670 Camera (Video) Example
 * Author: Khaled Magdy
 * For More Info Visit: www.DeepBlueMbedded.com
 */
#include <Arduino.h>
#include <Wire.h>
#include "OV7670_UART.h"

#define SCCB_SCANNER_DIAGNOSTIC 0

#if SCCB_SCANNER_DIAGNOSTIC
static void startCameraClock() {
  pinMode(3, OUTPUT);
  TCCR2A = _BV(COM2B1) | _BV(WGM21) | _BV(WGM20);
  TCCR2B = _BV(WGM22) | _BV(CS20);
  OCR2A = 1;
  OCR2B = 0;
}

static void scanSccbBus() {
  byte found = 0;
  Serial.print("SDA(A4)=");
  Serial.print(digitalRead(A4) ? "HIGH" : "LOW");
  Serial.print(" SCL(A5)=");
  Serial.println(digitalRead(A5) ? "HIGH" : "LOW");
  Serial.println("Scanning SCCB/I2C bus...");
  for (byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    byte error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("ACK at 0x");
      if (address < 16) {
        Serial.print("0");
      }
      Serial.println(address, HEX);
      found++;
    }
  }
  if (!found) {
    Serial.println("No SCCB/I2C devices acknowledged.");
  }
}
#endif

void setup() {
#if SCCB_SCANNER_DIAGNOSTIC
  Serial.begin(115200);
  delay(250);
  startCameraClock();
  delay(1000);
  pinMode(A4, INPUT_PULLUP);
  pinMode(A5, INPUT_PULLUP);
  Wire.begin();
  Wire.setClock(10000);
  scanSccbBus();
#else
  initializeScreenAndCamera();
  pinMode(8, INPUT_PULLUP);
#endif
}

void loop() {
#if SCCB_SCANNER_DIAGNOSTIC
  delay(2000);
  scanSccbBus();
#else
  processFrame();
#endif
}
