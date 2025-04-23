#pragma once
#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <string>
#include <vector>

class Image
{
public:
	Image(int width, int height);
	virtual ~Image();
	void setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);
	void writeToFile(const std::string &filename);
	int getWidth() const { return width; }
	int getHeight() const { return height; }
	// For CUDA return a pointer to the pixel data
	unsigned char *getPixels() { return &pixels[0]; }

private:
	int width;
	int height;
	int comp;
	std::vector<unsigned char> pixels;
};

#endif
