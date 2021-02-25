//Author: Daniil Pavlov

#include "CudaDomainDecomposition.h"
#include "CudaKernelSources.h"

using namespace OpenMM;
using namespace std;

CudaDDCalcForcesAndEnergyKernel::CudaDDCalcForcesAndEnergyKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : CalcForcesAndEnergyKernel(name, platform), data(data) {
    //TODO
}

void CudaDDCalcForcesAndEnergyKernel::initialize(const System& system) {
    //TODO
}

void CudaDDCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    //TODO
}

double CudaDDCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    //TODO
}


CudaDDUpdateStateDataKernel::CudaDDUpdateStateDataKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : UpdateStateDataKernel(name, platform), data(data) {
    //TODO
}

void CudaDDUpdateStateDataKernel::initialize(const System& system) {
    //TODO
}

double CudaDDUpdateStateDataKernel::getTime(const ContextImpl& context) const {
    //TODO
}

void CudaDDUpdateStateDataKernel::setTime(ContextImpl& context, double time) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getPositions(ContextImpl& context, vector<Vec3>& positions) {
    //TODO
}

void CudaDDUpdateStateDataKernel::setPositions(ContextImpl& context, const vector<Vec3>& positions) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getVelocities(ContextImpl& context, vector<Vec3>& velocities) {
    //TODO
}

void CudaDDUpdateStateDataKernel::setVelocities(ContextImpl& context, const vector<Vec3>& velocities) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getForces(ContextImpl& context, vector<Vec3>& forces) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getEnergyParameterDerivatives(ContextImpl& context, map<string, double>& derivs) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const {
    //TODO
}

void CudaDDUpdateStateDataKernel::setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c) {
    //TODO
}

void CudaDDUpdateStateDataKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    //TODO
}

void CudaDDUpdateStateDataKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    //TODO
}


CudaDDApplyConstraintsKernel::CudaDDApplyConstraintsKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : ApplyConstraintsKernel(name, platform), data(data) {
    //TODO
}

void CudaDDApplyConstraintsKernel::initialize(const System& system) {
    //TODO
}

void CudaDDApplyConstraintsKernel::apply(ContextImpl& context, double tol) {
    //TODO
}

void CudaDDApplyConstraintsKernel::applyToVelocities(ContextImpl& context, double tol) {
    //TODO
}

CudaDDVirtualSitesKernel::CudaDDVirtualSitesKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : VirtualSitesKernel(name, platform), data(data) {
    //TODO
}

void CudaDDVirtualSitesKernel::initialize(const System& system) {
    //TODO
}

void CudaDDVirtualSitesKernel::computePositions(ContextImpl& context) {
    //TODO
}


CudaDDCalcNonbondedForceKernel::CudaDDCalcNonbondedForceKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data, const System& system) : CalcNonbondedForceKernel(name, platform), data(data) {
    //TODO
}

void CudaDDCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    NonbondedMethod nonbondedMethod = CalcNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
    if(nonbondedMethod == PME || nonbondedMethod == LJPME)
        throw OpenMMException("PME for domain decomposition is not implemented yet.");
    //TODO
}

double CudaDDCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    //TODO
}

void CudaDDCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force) {
    //TODO
}

void CudaDDCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    throw OpenMMException("getPMEParametersInContext: PME for domain decomposition is not implemented yet.");
}

void CudaDDCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    throw OpenMMException("getPMEParametersInContext: PME for domain decomposition is not implemented yet.");
}


CudaDDIntegrateVerletStepKernel::CudaDDIntegrateVerletStepKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : IntegrateVerletStepKernel(name, platform), data(data) {
    //TODO
}

void CudaDDIntegrateVerletStepKernel::initialize(const System& system, const VerletIntegrator& integrator) {
    //TODO
}

void CudaDDIntegrateVerletStepKernel::execute(ContextImpl& context, const VerletIntegrator& integrator) {
    //TODO
}

double CudaDDIntegrateVerletStepKernel::computeKineticEnergy(ContextImpl& context, const VerletIntegrator& integrator) {
    //TODO
}