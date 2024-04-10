// Import the express router as shown in the lecture code
// Note: please do not forget to export the router!
import {Router} from 'express';
const router = Router();
import { products } from '../config/mongoCollections.js';
import * as validation from "../helpers.js"
import { create, get, remove, update } from '../data/products.js' 

router
  .route('/')
  .get(async (req, res) => {
    try {
      const productCollection = await products();
      let productList = await productCollection.find({}, {projection: {productName: 1}}).toArray();
      return res.json(productList);
    } catch (e) {
      return res.status(500).send(e);
  }
  })
  .post(async (req, res) => {
    const postProducts = req.body;
    if (!postProducts || Object.keys(postProducts).length === 0) {
      return res
        .status(400)
        .json({error: 'There are no fields in the request body'});
    }
    try {
      postProducts.productName = validation.checkString(postProducts.productName, 'Name');
      postProducts.productDescription = validation.checkString(postProducts.productDescription, 'Description');
      postProducts.modelNumber = validation.checkString(postProducts.modelNumber, 'Model Number');
      validation.checkNumber(postProducts.price, "Price");
      if(postProducts.price != postProducts.price.toFixed(2)) throw "price must be at most 2 decimal places"
      postProducts.manufacturer = validation.checkString(postProducts.manufacturer, "Manufacturer");
      postProducts.manufacturerWebsite = validation.checkString(postProducts.manufacturerWebsite, "manufacturerWebsite");
      if(postProducts.manufacturerWebsite.substring(postProducts.manufacturerWebsite.length-4) != ".com") throw "manufacturerWebsite must end with .com";
      if(postProducts.manufacturerWebsite.substring(0, 11) != "http://www.") throw "manufacturerWebsite must begin with https://www.";
      if(postProducts.manufacturerWebsite.length <  20) throw "domain name must be 5 or more characters"
      validation.checkStringArray(postProducts.keywords, "keywords");
      validation.checkStringArray(postProducts.categories, "categories");
      postProducts.dateReleased = validation.checkDate(postProducts.dateReleased);
      if(!postProducts.discontinued && typeof postProducts.discontinued != 'boolean') throw "discontinued must be a boolean";
    } catch (e) {
      return res.status(400).json({error: e});
    }

    try {
      const newPost = await create(
        postProducts.productName,
        postProducts.productDescription,
        postProducts.modelNumber,
        postProducts.price,
        postProducts.manufacturer,
        postProducts.manufacturerWebsite,
        postProducts.keywords,
        postProducts.categories,
        postProducts.dateReleased,
        postProducts.discontinued
      );
      return res.json(newPost);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  });

router
  .route('/:productId')
  .get(async (req, res) => {
    try {
      req.params.productId = validation.checkID(req.params.productId, 'Id URL Param');
    } catch (e) {
      return res.status(400).json({error: e});
    }

    try {
      const post = await get(req.params.productId.toString());
      return res.status(200).json(post);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  })
  .delete(async (req, res) => {
    try {
      req.params.productId = validation.checkID(req.params.productId, 'Id URL Param');
    } catch (e) {
      return res.status(400).json(req.params.productId);
    }
    //try to delete post
    try {
      let deletedPost = await remove(req.params.productId.toString());
      return res.status(200).json(deletedPost);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  })
  .put(async (req, res) => {
    const postProducts = req.body;
    if (!postProducts || Object.keys(postProducts).length === 0) {
      return res
        .status(400)
        .json({error: 'There are no fields in the request body'});
    }
    try {
      req.params.productId = validation.checkID(req.params.productId, 'Id URL Param');
    } catch (e) {
      return res.status(400).json(req.params.productId);
    }
    try {
      postProducts.productName = validation.checkString(postProducts.productName, 'Name');
      postProducts.productDescription = validation.checkString(postProducts.productDescription, 'Description');
      postProducts.modelNumber = validation.checkString(postProducts.modelNumber, 'Model Number');
      validation.checkNumber(postProducts.price, "Price");
      if(postProducts.price != postProducts.price.toFixed(2)) throw "price must be at most 2 decimal places"
      postProducts.manufacturer = validation.checkString(postProducts.manufacturer, "Manufacturer");
      postProducts.manufacturerWebsite = validation.checkString(postProducts.manufacturerWebsite, "manufacturerWebsite");
      if(postProducts.manufacturerWebsite.substring(postProducts.manufacturerWebsite.length-4) != ".com") throw "manufacturerWebsite must end with .com";
      if(postProducts.manufacturerWebsite.substring(0, 11) != "http://www.") throw "manufacturerWebsite must begin with https://www.";
      if(postProducts.manufacturerWebsite.length <  20) throw "domain name must be 5 or more characters"
      validation.checkStringArray(postProducts.keywords, "keywords");
      validation.checkStringArray(postProducts.categories, "categories");
      postProducts.dateReleased = validation.checkDate(postProducts.dateReleased);
      if(!postProducts.discontinued && typeof postProducts.discontinued != 'boolean') throw "discontinued must be a boolean";
    } catch (e) {
      return res.status(400).json({error: e});
    }

    try {
      const updatedPost = await update(
        req.params.productId.toString(),
        postProducts.productName,
        postProducts.productDescription,
        postProducts.modelNumber,
        postProducts.price,
        postProducts.manufacturer,
        postProducts.manufacturerWebsite,
        postProducts.keywords,
        postProducts.categories,
        postProducts.dateReleased,
        postProducts.discontinued
      );
      return res.status(200).json(updatedPost);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  });

export default router;