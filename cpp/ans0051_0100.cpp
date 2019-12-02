/*
* @Author: Chen Zhiquan
* @Date:   2019-12-02 14:36:37
* @Last Modified by:   octavian
* @Last Modified time: 2019-12-02 15:40:16
*/
//98 Validate Binary Search Tree
bool isValidate(TreeNode* root){
    stack <TreeNode*> st;

    TreeNode *p = root, *pre = NULL;

    while (!st.empty() || p){
        while (p){
            st.push(p);
            p = p->left;
        }

        p = st.top();
        st.pop();

        if (pre && p->val <= pre->val) return false;

        pre = p;
        p = p->right;
    }
    return true;
}

// 99 Recover Binary Search Tree
void recoverTree(TreeNode* root){
    
}