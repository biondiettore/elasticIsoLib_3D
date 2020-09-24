#ifndef STAGGER_3D_H
#define STAGGER_3D_H 1

#include <float3DReg.h>
#include <operator.h>

using namespace SEP;

//! Interpolate a grid of values up or down one half grid cell
/*!
 Used for creating the staggered grid elastic parameters used for leastic wave prop
*/
class staggerZ : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:

    int _nz;
    int _nx;
		int _ny;

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		staggerZ(const std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data);

    //! FWD
		/*!
    * this interpolates a grid of values 1/2 grid point down
    */
    void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;

    //! ADJ
    /*!
    * this interpolates a grid of values 1/2 grid point up
    */
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		//! Desctructor
    /*!
    * A more elaborate description of Destructor
    */
		~staggerZ(){};

};

//! Interpolate a grid to the left or right one half grid cell along the x axis
/*!
 Used for creating the staggered grid elastic parameters used for leastic wave prop
*/
class staggerX : public Operator<float3DReg, float3DReg> {

	private:

    int _nz;
		int _nx;
		int _ny;

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		staggerX(const std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data);

    //! FWD
		/*!
    * this interpolates a grid of values 1/2 grid point to the right
    */
    void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;

    //! ADJ
    /*!
    * this interpolates a grid of values 1/2 grid point to the left
    */
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		//! Desctructor
    /*!
    * A more elaborate description of Destructor
    */
		~staggerX(){};

};

//! Interpolate a grid of values left or right one half grid cell along the y axis
/*!
 Used for creating the staggered grid elastic parameters used for leastic wave prop
*/
class staggerY : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:

    int _nz;
    int _nx;
		int _ny;

	public:
    //! Constructor.
		/*!
    * Overloaded constructors from operator
    */
		staggerY(const std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data);

    //! FWD
		/*!
    * this interpolates a grid of values 1/2 grid point down
    */
    void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;

    //! ADJ
    /*!
    * this interpolates a grid of values 1/2 grid point up
    */
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		//! Desctructor
    /*!
    * A more elaborate description of Destructor
    */
		~staggerY(){};

};

#endif
