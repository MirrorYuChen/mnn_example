#include "detecter.h"
#include "./centerface/centerface.h"
#include "./ultraface/ultraface.h"

namespace mirror {
Detecter* CenterfaceFactory::CreateDetecter() {
    return new Centerface();
}

Detecter* UltrafaceFactory::CreateDetecter() {
    return new UltraFace();
}


}