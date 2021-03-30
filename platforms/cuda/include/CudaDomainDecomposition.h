#ifndef OPENMM_CUDADOMAINDECOMPOSITION_H_
#define OPENMM_CUDADOMAINDECOMPOSITION_H_

//Author: Daniil Pavlov

#include "CudaPlatform.h"
#include "CudaContext.h"
#include "CudaKernels.h"
#include "openmm/common/CommonKernels.h"

namespace OpenMM {

/**
 * This is a base class from which all domain decomposition kernels are derived.
 * It's used as an interface with kernels by CudaDDUtilities.
 * A kernel that implements this interface is assumed to exist for the whole duration of PlatformData
 * existence because it's permanently registered in CudaDDUtilities on construction.
 */
class CudaDDInterface {
public:
    CudaDDInterface(std::string name, const Platform& platform, CudaPlatform::PlatformData& data);
    /**
     * Destroys all the kernels to prepare for contexts destruction.
     */
    virtual void destroyKernels();
    /**
     * Reconstructs all the kernels if there are none.
     */
    virtual void prepareKernels();
protected:
    std::string name;
    const Platform& platform;
    CudaPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

class CudaDDUpdateStateDataKernel;

/**
 * This class implements domain decomposition features. This one is special because it's stored in
 * PlatformData rather than CudaContext.
 */
class OPENMM_EXPORT_COMMON CudaDDUtilities {
    friend class CudaDDUpdateStateDataKernel;
public:
    struct Domain {
        double xlo, xhi, ylo, yhi, zlo, zhi;
    };
    CudaDDUtilities(CudaPlatform::PlatformData& data, const System& system, ContextImpl& contextImpl);
    /**
     * Register a kernel via CudaDDInterface.
     * 
     * @param kernel reference to a kernel.
     */
    void registerKernel(CudaDDInterface* kernel);
    /**
     * Register the updater kernel.
     * 
     * @param kernel reference to a kernel.
     */
    void registerUpdater(CudaDDUpdateStateDataKernel* kernel);
    /**
     * Create CudaContexts if there aren't any. Particle positions need to be set in order for this to work.
     * This is needed because in domain decomposition mode CudaContext creation is delegated to CudaDDUtilities.
     */
    void prepareContexts();
    /**
     * Destroys all CudaContexts. PlatformData holds ownership over contexts so this needn't be called on destruction.
     */
    void destroyContexts();
    /**
     * Perform the domain decomposition.
     */
    void decompose();
    /**
     * This should be called whenever cutoff of a force changes or might have changed.
     */
    void resetCutoff();
    /**
     * Get the maximum cutoff distance used by any force group.
     */
    double getCutoff();
private:
    int numAtoms, paddedNumAtoms;

    CudaPlatform::PlatformData& data;
    std::vector<std::vector<int> > molecules;
    std::vector<int> moleculeInd;
    std::vector<CudaDDInterface*> registeredKernels;
    CudaDDUpdateStateDataKernel* updater;

    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    Vec3 box[3];
    double time;

    double cutoff;
    std::vector<std::vector<unsigned int> > domainMasks;
    std::vector<std::vector<unsigned int> > enabledMasks;
    std::vector<Domain> domains;
    std::vector<int> domainInd;
public:
    const System& system;
    ContextImpl& contextImpl;
};

/**
 * This kernel is invoked at the beginning and end of force and energy computations.  It gives the
 * Platform a chance to clear buffers and do other initialization at the beginning, and to do any
 * necessary work at the end to determine the final results.
 */
class CudaDDCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel, public CudaDDInterface {
public:
    CudaDDCalcForcesAndEnergyKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data) :
         CalcForcesAndEnergyKernel(name, platform), CudaDDInterface(name, platform, data) {
    }
    CudaCalcForcesAndEnergyKernel& getKernel(int i) {
        return kernels[i].getAs<CudaCalcForcesAndEnergyKernel>();
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system) {
    }
    /**
     * This is called at the beginning of each force/energy computation, before calcForcesAndEnergy() has been called on
     * any ForceImpl.
     *
     * @param context       the context in which to execute this kernel
     * @param includeForce  true if forces should be computed
     * @param includeEnergy true if potential energy should be computed
     * @param groups        a set of bit flags for which force groups to include
     */
    void beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups);
    /**
     * This is called at the end of each force/energy computation, after calcForcesAndEnergy() has been called on
     * every ForceImpl.
     *
     * @param context       the context in which to execute this kernel
     * @param includeForce  true if forces should be computed
     * @param includeEnergy true if potential energy should be computed
     * @param groups        a set of bit flags for which force groups to include
     * @param valid         the method may set this to false to indicate the results are invalid and the force/energy
     *                      calculation should be repeated
     * @return the potential energy of the system.  This value is added to all values returned by ForceImpls'
     * calcForcesAndEnergy() methods.  That is, each force kernel may <i>either</i> return its contribution to the
     * energy directly, <i>or</i> add it to an internal buffer so that it will be included here.
     */
    double finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid);
};


/**
 * This kernel provides methods for setting and retrieving various state data: time, positions,
 * velocities, and forces. Unlike other DD classes, this one doesn't implement CudaDDInterface.
 */
class CudaDDUpdateStateDataKernel : public UpdateStateDataKernel, public CudaDDInterface {
public:
    CudaDDUpdateStateDataKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data);
    CudaUpdateStateDataKernel& getKernel(int i) {
        return kernels[i].getAs<CudaUpdateStateDataKernel>();
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system) {
    }
    /**
     * Get the current time (in picoseconds).
     *
     * @param context    the context in which to execute this kernel
     */
    double getTime(const ContextImpl& context) const;
    /**
     * Set the current time (in picoseconds).
     *
     * @param context    the context in which to execute this kernel
     */
    void setTime(ContextImpl& context, double time);
    /**
     * Get the positions of all particles.
     *
     * @param positions  on exit, this contains the particle positions
     */
    void getPositions(ContextImpl& context, std::vector<Vec3>& positions);
    /**
     * Set the positions of all particles.
     *
     * @param positions  a vector containg the particle positions
     */
    void setPositions(ContextImpl& context, const std::vector<Vec3>& positions);
    /**
     * Get the velocities of all particles.
     *
     * @param velocities  on exit, this contains the particle velocities
     */
    void getVelocities(ContextImpl& context, std::vector<Vec3>& velocities);
    /**
     * Set the velocities of all particles.
     *
     * @param velocities  a vector containg the particle velocities
     */
    void setVelocities(ContextImpl& context, const std::vector<Vec3>& velocities);
    /**
     * Get the current forces on all particles.
     *
     * @param forces  on exit, this contains the forces
     */
    void getForces(ContextImpl& context, std::vector<Vec3>& forces);
    /**
     * Get the current derivatives of the energy with respect to context parameters.
     *
     * @param derivs  on exit, this contains the derivatives
     */
    void getEnergyParameterDerivatives(ContextImpl& context, std::map<std::string, double>& derivs);
    /**
     * Get the current periodic box vectors.
     *
     * @param a      on exit, this contains the vector defining the first edge of the periodic box
     * @param b      on exit, this contains the vector defining the second edge of the periodic box
     * @param c      on exit, this contains the vector defining the third edge of the periodic box
     */
    void getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const;
    /**
     * Set the current periodic box vectors.
     *
     * @param a      the vector defining the first edge of the periodic box
     * @param b      the vector defining the second edge of the periodic box
     * @param c      the vector defining the third edge of the periodic box
     */
    void setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c);
    /**
     * Create a checkpoint recording the current state of the Context.
     * 
     * @param stream    an output stream the checkpoint data should be written to
     */
    void createCheckpoint(ContextImpl& context, std::ostream& stream);
    /**
     * Load a checkpoint that was written by createCheckpoint().
     * 
     * @param stream    an input stream the checkpoint data should be read from
     */
    void loadCheckpoint(ContextImpl& context, std::istream& stream);
private:
    CudaPlatform::PlatformData& data;
};


/**
 * This kernel modifies the positions of particles to enforce distance constraints.
 */
class CudaDDApplyConstraintsKernel : public ApplyConstraintsKernel, public CudaDDInterface {
public:
    CudaDDApplyConstraintsKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data) :
        ApplyConstraintsKernel(name, platform), CudaDDInterface(name, platform, data) {
    }
    CommonApplyConstraintsKernel& getKernel(int i) {
        return kernels[i].getAs<CommonApplyConstraintsKernel>();
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system) {
    }
    /**
     * Update particle positions to enforce constraints.
     *
     * @param context    the context in which to execute this kernel
     * @param tol        the distance tolerance within which constraints must be satisfied.
     */
    void apply(ContextImpl& context, double tol);
    /**
     * Update particle velocities to enforce constraints.
     *
     * @param context    the context in which to execute this kernel
     * @param tol        the velocity tolerance within which constraints must be satisfied.
     */
    void applyToVelocities(ContextImpl& context, double tol);
};

/**
 * This kernel recomputes the positions of virtual sites.
 */
class CudaDDVirtualSitesKernel : public VirtualSitesKernel, public CudaDDInterface {
public:
    CudaDDVirtualSitesKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data) :
        VirtualSitesKernel(name, platform), CudaDDInterface(name, platform, data) {
    }
    CommonVirtualSitesKernel& getKernel(int i) {
        return kernels[i].getAs<CommonVirtualSitesKernel>();
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system) {
    }
    /**
     * Compute the virtual site locations.
     *
     * @param context    the context in which to execute this kernel
     */
    void computePositions(ContextImpl& context);
};

/**
 * This kernel is invoked by NonbondedForce to calculate the forces acting on the system.
 */
class CudaDDCalcNonbondedForceKernel : public CalcNonbondedForceKernel, public CudaDDInterface {
public:
    CudaDDCalcNonbondedForceKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data, const System& system) :
        CalcNonbondedForceKernel(name, platform), CudaDDInterface(name, platform, data), storedForce(nullptr) {
    }
    CudaCalcNonbondedForceKernel& getKernel(int i) {
        return kernels[i].getAs<CudaCalcNonbondedForceKernel>();
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the NonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const NonbondedForce& force) {
        storedForce = &force;
    }
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeDirect  true if direct space interactions should be included
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the NonbondedForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const NonbondedForce& force);
    /**
     * Get the parameters being used for PME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the dispersion parameters being used for the dispersion term in LJPME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void prepareKernels() override;
private:
    const NonbondedForce* storedForce;
};

/**
 * This kernel is invoked by VerletIntegrator to take one time step.
 */
class CudaDDIntegrateVerletStepKernel : public IntegrateVerletStepKernel, public CudaDDInterface {
public:
    CudaDDIntegrateVerletStepKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data) :
        IntegrateVerletStepKernel(name, platform), CudaDDInterface(name, platform, data), storedIntegrator(nullptr) {
    }
    CommonIntegrateVerletStepKernel& getKernel(int i) {
        return kernels[i].getAs<CommonIntegrateVerletStepKernel>();
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the VerletIntegrator this kernel will be used for
     */
    void initialize(const System& system, const VerletIntegrator& integrator) {
        storedIntegrator = &integrator;
    }
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VerletIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const VerletIntegrator& integrator);
    /**
     * Compute the kinetic energy.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the VerletIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const VerletIntegrator& integrator);
    void prepareKernels() override;
private:
    const VerletIntegrator* storedIntegrator;
};

} // namespace OpenMM

#endif // OPENMM_CUDADOMAINDECOMPOSITION_H_