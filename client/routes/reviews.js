// Import the express router as shown in the lecture code
// Note: please do not forget to export the router!
import {Router} from 'express';
const router = Router();
import {createReview, getAllReviews, getReview, updateReview, removeReview} from "../data/reviews.js";
import * as validation from "../helpers.js"
import { products } from '../config/mongoCollections.js';

router
  .route('/:productId')
  .get(async (req, res) => {
    try {
      req.params.productId = validation.checkID(req.params.productId, 'Id URL Param');
    } catch (e) {
      return res.status(400).json({error: e});
    }

    try {
      const post = await getAllReviews(req.params.productId.toString());
      return res.status(200).json(post);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  })
  .post(async (req, res) => {
    const postReviews = req.body;
    if (!postReviews || Object.keys(postReviews).length === 0) {
      return res
        .status(400)
        .json({error: 'There are no fields in the request body'});
    }
    try{
      req.params.productId = validation.checkID(req.params.productId, 'Id URL Param');
      postReviews.title = validation.checkString(postReviews.title, "title");
      postReviews.reviewerName = validation.checkString(postReviews.reviewerName, "reviewerName");
      postReviews.review = validation.checkString(postReviews.review, "reviewer");
      validation.checkNumber(postReviews.rating, "rating");
      if(postReviews.rating < 1 || postReviews.rating > 5) throw "rating must be between 1-5"
      if(postReviews.rating != postReviews.rating.toFixed(1)) throw "rating must be at most 1 decimal place" 
    }catch(e){
      return res.status(400).json({eprror: e});
    }

    try {
      const newReview = await createReview(
        req.params.productId.toString(),
        postReviews.title,
        postReviews.reviewerName,
        postReviews.review,
        postReviews.rating
      );
      const productCollection = await products();
      let product = await productCollection.findOne({"reviews._id": newReview._id});
      return res.json(product);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  });

router
  .route('/review/:reviewId')
  .get(async (req, res) => {
    try {
      req.params.reviewId = validation.checkID(req.params.reviewId, 'Id URL Param');
    } catch (e) {
      return res.status(400).json({error: e});
    }

    try {
      const post = await getReview(req.params.reviewId.toString());
      return res.status(200).json(post);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  })
  .patch(async (req, res) => {
    const requestBody = req.body;
    //check to make sure there is something in req.body
    if (!requestBody || Object.keys(requestBody).length === 0) {
      return res
        .status(400)
        .json({error: 'There are no fields in the request body'});
    }
    //check the inputs that will return 400 is fail
    try {
      req.params.reviewId = validation.checkID(req.params.reviewId, 'Id URL Param');
      if(requestBody.title) requestBody.title = validation.checkString(requestBody.title, "title");
      if(requestBody.reviewerName) requestBody.reviewerName = validation.checkString(requestBody.reviewerName, "reviewerName");
      if(requestBody.review) requestBody.review = validation.checkString(requestBody.review, "reviewer");
      if(requestBody.rating != undefined){
        validation.checkNumber(requestBody.rating, "rating");
        if(requestBody.rating < 1 ||requestBody.rating > 5) throw "rating must be between 1-5"
        if(requestBody.rating !=requestBody.rating.toFixed(1)) throw "rating must be at most 1 decimal place" 
      }
    } catch (e) {
      return res.status(400).json({error: e});
    }
    //try to perform update
    try {
      const updatedPost = await updateReview(req.params.reviewId.toString(), requestBody);
      return res.status(200).json(updatedPost);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  })
  .delete(async (req, res) => {
    try {
      req.params.reviewId = validation.checkID(req.params.reviewId, 'Id URL Param');
    } catch (e) {
      return res.status(400).json({error: e});
    }
    //try to delete post
    try {
      let deletedPost = await removeReview(req.params.reviewId.toString());
      return res.status(200).json(deletedPost);
    } catch (e) {
      return res.status(404).json({error: e});
    }
  });

export default router;