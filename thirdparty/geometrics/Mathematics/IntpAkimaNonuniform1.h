// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2021
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 4.0.2019.08.13

#pragma once

#include "IntpAkima1.h"
#include "core/error_macros.h"

template <typename Real>
class IntpAkimaNonuniform1 : public InterpolationAkima1<Real>
{
public:
    // Construction.  The interpolator is for arbitrarily spaced x-values.
    // The input arrays must have 'quantity' elements and the X[] array
    // must store increasing values:  X[i + 1] > X[i] for all i.
    IntpAkimaNonuniform1(int quantity, Real const* X, Real const* F)
        :
        InterpolationAkima1<Real>(quantity, F),
        mX(X)
    {
        ERR_FAIL_NULL_MSG(X, "Invalid input.");
        for (int j0 = 0, j1 = 1; j1 < quantity; ++j0, ++j1)
        {
            if (X[j1] < X[j0]) 
            {
                ERR_PRINT_ONCE(vformat("Invalid input X[i + 1] %f is smaller than X[i] %f",  X[j1], X[j0]));
                return;
            }
        }

        // Compute slopes.
        std::vector<Real> slope(quantity + 3);
        int i, ip1, ip2;
        for (i = 0, ip1 = 1, ip2 = 2; i < quantity - 1; ++i, ++ip1, ++ip2)
        {
            Real dx = X[ip1] - X[i];
            Real df = F[ip1] - F[i];
            slope[ip2] = df / dx;
        }

        slope[1] = (Real)2 * slope[2] - slope[3];
        slope[0] = (Real)2 * slope[1] - slope[2];
        slope[quantity + 1] = (Real)2 * slope[quantity] - slope[quantity - 1];
        slope[quantity + 2] = (Real)2 * slope[quantity + 1] - slope[quantity];

        // Construct derivatives.
        Vector<Real> FDer;
        FDer.resize(quantity);
        for (i = 0; i < quantity; ++i)
        {
            FDer.write[i] = this->ComputeDerivative(&slope[i]);
        }

        // Construct polynomials.
        for (i = 0, ip1 = 1; i < quantity - 1; ++i, ++ip1)
        {
            auto& poly = this->mPoly.write[i];

            Real F0 = F[i];
            Real F1 = F[ip1];
            Real FDer0 = FDer[i];
            Real FDer1 = FDer[ip1];
            Real df = F1 - F0;
            Real dx = X[ip1] - X[i];
            Real dx2 = dx * dx;
            Real dx3 = dx2 * dx;

            poly[0] = F0;
            poly[1] = FDer0;
            poly[2] = ((Real)3 * df - dx * (FDer1 + (Real)2 * FDer0)) / dx2;
            poly[3] = (dx * (FDer0 + FDer1) - (Real)2 * df) / dx3;
        }
    }

    virtual ~IntpAkimaNonuniform1() = default;

    // Member access.
    Real const* GetX() const
    {
        return mX;
    }

    virtual Real GetXMin() const override
    {
        return mX[0];
    }

    virtual Real GetXMax() const override
    {
        return mX[this->mQuantity - 1];
    }

protected:
    virtual void Lookup(Real x, int& index, Real& dx) const override
    {
        // The caller has ensured that mXMin <= x <= mXMax.
        for (index = 0; index + 1 < this->mQuantity; ++index)
        {
            if (x < mX[index + 1])
            {
                dx = x - mX[index];
                return;
            }
        }

        --index;
        dx = x - mX[index];
    }

    Real const* mX;
};
