#pragma once // --> directive designed to cause the current source file to be included only once in a single compilation. Thus, #pragma once serves the same purpose as #include guards, but with several advantages, including: less code, avoiding name clashes, and improved compile speed
#include <cmath>
#include <iostream>
#include "Vector.h"
#include <iomanip>

template <class V1, class V2>
class Operator
{
	private:
		/****************************** Member variables ********************************/
  		std::shared_ptr<V1> _domain;
  		std::shared_ptr<V2> _range;

	protected:

		/********************************* Constructors *********************************/
  		Operator() { ; }

	public:

		/********************************* Deconstructors *******************************/
 		virtual ~Operator<V1, V2>() { ; }

		/********************************* Other functions ******************************/
  		virtual void forward(const bool add, const std::shared_ptr<V1> model, std::shared_ptr<V2> data) const = 0;
  		virtual void adjoint(const bool add, std::shared_ptr<V1> model, const std::shared_ptr<V2> data) const = 0;

		/********************************* Useful functions *****************************/

		/* Check domain and range of operator */
  		virtual bool checkDomainRange(const std::shared_ptr<V1> model, const std::shared_ptr<V2> data) const
  		{
			bool ret = true;

			if (!_domain->checkSame(model))
			{
				std::cerr << "Domains do not match" << std::endl;
				ret = false;
			}
			if (!_range->checkSame(data))
			{
				std::cerr << "Ranges do not match" << std::endl;
				ret = false;
			}
			return ret;
  		}

		/* Dot product test */
		bool dotTest(const bool verbose = false, const double maxError = .00001) const
		{
			std::cout << "dot Test 1" << std::endl;
			std::shared_ptr<V1> d1 = _domain->clone(), d2 = _domain->clone();
			std::shared_ptr<V2> r1 = _range->clone(), r2 = _range->clone();
			d1->random();
			r1->random();
			std::cout << "dot Test 2" << std::endl;
			/* Compute dot product WITHOUT "add" */

			adjoint(false, d2, r1);
			forward(false, d1, r2);
			std::cout << "dot Test 3" << std::endl;
			double dot1 = r1->dot(r2), dot2 = d1->dot(d2);
			double errorNoAdd = fabs((dot1 - dot2)/dot2); /* fabs is absolute value for double */
			if(verbose)
			{
				std::cout << "WITHOUT ADD: " << std::endl;
				std::cout << "dot1: " << std::setprecision(16) << dot1 << std::endl;
				std::cout << "dot2: " << std::setprecision(16) << dot2 << std::endl;
				std::cout << "error WITHOUT ADD: " << std::setprecision(16) << errorNoAdd << std::endl;
			}

			/* Compute dot product WITH "add" */
			forward(true, d1, r2);
			adjoint(true, d2, r1);
			double dot3 = r1->dot(r2), dot4 = d1->dot(d2);
			double errorAdd = fabs((dot3 - dot4)/dot4);
			if(verbose)
			{
				std::cout << "WITH ADD: " << std::endl;
				std::cout << "dot3: " << std::setprecision(16) << dot3 << std::endl;
				std::cout << "dot4: " << std::setprecision(16) << dot4 << std::endl;
				std::cout << "error with ADD: " << std::setprecision(16) << errorAdd << std::endl;
			}

			if ( errorNoAdd > maxError) return false;
			if ( errorAdd > maxError) return false;

			return true;
		}

		/********************************* Modifiers *************************************/
		// Set domain and range of operator
  		void setDomainRange(const std::shared_ptr<V1> dom, const std::shared_ptr<V2> rang){
    		_domain = dom->cloneSpace();
    		_range = rang->cloneSpace();
  		}

		/******************************** Accessors *************************************/
		std::shared_ptr<V1> getDomain() { return _domain; }
  		std::shared_ptr<V2> getRange() { return _range; }
};
