// This file will import both route files and export the constructor method as shown in the lecture code

/*
    - When the route is /products use the routes defined in the products.js routing file
    - When the route is /reviews use the routes defined in reviews.js routing file
    - All other enpoints should respond with a 404 as shown in the lecture code
*/
import productsRoutes from './products.js';
import reviewsRoutes from './reviews.js'

const constructorMethod = (app) => {
  app.use('/products', productsRoutes);
  app.use('/reviews', reviewsRoutes);


  app.use('*', (req, res) => {
    return res.status(404).json({error: '404: womp womp'});
  });
};

export default constructorMethod;
